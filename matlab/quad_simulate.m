function log = quad_simulate(cfg, mode, trajectory, progress)
if nargin < 4
    progress = [];
end
if ~ismember(mode, {'pid', 'eso', 'ceso', 'tanh', 'td3'})
    error('Unknown controller.');
end

dt = cfg.dt;
n = round(cfg.duration/dt)+1;
if n < 2
    error('quad:Duration', 'Duration must be at least one timestep.');
end

ctrl = controller_init(cfg, mode);
state = initial_state(cfg);

log.time = (0:n-1)'*dt;
log.state = zeros(n, 17);
log.euler = zeros(n, 3);
log.reference = zeros(n, 3);
log.attitude_reference = zeros(n, 3);
log.estimate = zeros(n, 6);
log.target = zeros(n, 4);
log.force = zeros(n, 3);
log.saturated = false(n, 1);
log.gains = zeros(n, 3);

for k = 1:n
    t = log.time(k);
    [target, ctrl, out] = controller_step(ctrl, cfg, mode, t, state, trajectory);

    log.state(k,:) = state';
    log.euler(k,:) = quad_math('euler', state(7:10))';
    log.reference(k,:) = out.reference';
    log.attitude_reference(k,:) = out.attitude';
    log.estimate(k,:) = out.estimate';
    log.target(k,:) = target';
    log.force(k,:) = quad_disturbance(t, cfg.enable_disturbance)';
    log.saturated(k) = ctrl.saturated;
    log.gains(k,:) = ctrl.attWc';

    if k < n
        state = rk4_step(t, state, target, cfg, ctrl.mixer);
        if any(~isfinite(state)) || state(3) < 0.1 || norm(state(1:3)) > 50
            error('quad:Envelope', 'Flight envelope exceeded at %.3f s (%s, %s).', t, mode, trajectory);
        end
    end

    if ~isempty(progress) && (mod(k, 500) == 0 || k == n)
        progress(k/n);
    end
end

log.metrics = flight_metrics(log, dt);
log.config = cfg;
log.mode = mode;
log.trajectory = trajectory;
end


function ctrl = controller_init(cfg, mode)
dt = cfg.dt;
levels = 1 + strcmp(mode, 'ceso');
ctrl.banks = {observer_bank(cfg.position_observer_bandwidth, levels, cfg.cascade_ratio, dt), ...
              observer_bank(cfg.attitude_observer_bandwidth, levels, cfg.cascade_ratio, dt)};
ctrl.nl = {tanh_adrc_init(cfg, 'position', false), tanh_adrc_init(cfg, 'attitude', true)};

ctrl.td3 = cfg.(['td3_' cfg.td3_variant]);
ctrl.policy = [];
ctrl.period = round(ctrl.td3.agent_period/dt);
ctrl.maxStep = ctrl.td3.max_gain_rate*dt;
ctrl.count = 0;
ctrl.obs = zeros(12+3*ctrl.td3.observe_gains, 1);
ctrl.attIntegral = zeros(3, 1);
ctrl.attWc = cfg.attitude_bandwidth(:);
ctrl.wcTarget = ctrl.attWc;

if strcmp(mode, 'td3')
    file = fullfile(fileparts(fileparts(mfilename('fullpath'))), ctrl.td3.policy);
    if ~isfile(file)
        error('quad:Policy', 'No TD3 policy at %s. Train one with python/td3_train.py.', file);
    end
    policy = jsondecode(fileread(file));
    ctrl.policy = policy.layers;
end

ctrl.lastError = zeros(2, 3);
ctrl.derivative = zeros(2, 3);
ctrl.integral = zeros(2, 3);
ctrl.initialized = false(2, 1);
ctrl.saturated = false;
ctrl.noiseState = cfg.seed;

a = cfg.arm_length/sqrt(2);
xy = [a a;-a a;-a -a;a -a];
ctrl.mixer = [ones(1,4);xy(:,2)';-xy(:,1)';cfg.yaw_moment_per_thrust*[1 -1 1 -1]];
ctrl.invMixer = inv(ctrl.mixer);
end


function state = initial_state(cfg)
state = zeros(17, 1);
state(1:3) = cfg.initial_position(:);
state(7) = 1;
state(14:17) = cfg.mass*cfg.gravity/4;
end


function [target, ctrl, out] = controller_step(ctrl, cfg, mode, t, state, trajectory)
if strcmp(mode, 'td3')
    ctrl = update_gains(ctrl);
end
ctrl.count = ctrl.count+1;

[measuredP, measuredA, rmat, ctrl] = measure(ctrl, cfg, state);
[desiredP, desiredYaw] = quad_reference(t, trajectory);
wrench = ctrl.mixer*state(14:17);

[force, fp, ctrl] = position_loop(ctrl, cfg, mode, desiredP, measuredP, rmat, wrench);
desiredA = attitude_reference(force, desiredYaw);
[torque, fa, ctrl] = attitude_loop(ctrl, cfg, mode, desiredA, measuredA, wrench);

if strcmp(mode, 'td3')
    ctrl = observe(ctrl, cfg, desiredA, measuredA);
end

[target, ctrl] = allocate(ctrl, cfg, force, rmat, torque);
out.reference = desiredP;
out.attitude = desiredA;
out.estimate = [fp;fa];
end


function ctrl = update_gains(ctrl)
if mod(ctrl.count, ctrl.period) == 0
    ctrl = set_td3_action(ctrl, actor_forward(ctrl.policy, ctrl.obs));
end

if ctrl.maxStep > 0
    ctrl.attWc = ctrl.attWc+max(-ctrl.maxStep, min(ctrl.maxStep, ctrl.wcTarget-ctrl.attWc));
    ctrl.nl{2} = tanh_set_bandwidth(ctrl.nl{2}, ctrl.attWc);
end
end


function [measuredP, measuredA, rmat, ctrl] = measure(ctrl, cfg, state)
measuredA = quad_math('euler', state(7:10));
rmat = quad_math('rotation', state(7:10));
[ctrl.noiseState, noise] = sample_noise(ctrl.noiseState);

measuredP = state(1:3);
if cfg.enable_noise
    measuredP = measuredP + cfg.position_noise_std*noise(1:3);
    measuredA = measuredA + cfg.attitude_noise_std*noise(4:6);
end
end


function [force, fp, ctrl] = position_loop(ctrl, cfg, mode, desiredP, measuredP, rmat, wrench)
knownP = rmat(:,3)*wrench(1)/cfg.mass-[0;0;cfg.gravity];
[acc, fp, ctrl] = feedback(desiredP, measuredP, knownP, 1, ctrl, cfg, mode);

acc(3) = max(-cfg.max_vertical_acceleration, min(cfg.max_vertical_acceleration, acc(3)));
limit = min(cfg.max_horizontal_acceleration, (cfg.gravity+acc(3))*tan(deg2rad(cfg.max_tilt_deg)));
h = norm(acc(1:2));
if h > limit
    acc(1:2) = acc(1:2)*limit/h;
end

force = cfg.mass*(acc+[0;0;cfg.gravity]);
end


function desiredA = attitude_reference(force, yaw)
direction = force/norm(force);
phi = asin(max(-1, min(1, direction(1)*sin(yaw)-direction(2)*cos(yaw))));
theta = atan2(direction(1)*cos(yaw)+direction(2)*sin(yaw), direction(3));
desiredA = [phi;theta;yaw];
end


function [torque, fa, ctrl] = attitude_loop(ctrl, cfg, mode, desiredA, measuredA, wrench)
emap = quad_math('rate_map', measuredA);
knownA = emap*(wrench(2:4)./cfg.inertia(:));
[angularAcc, fa, ctrl] = feedback(desiredA, measuredA, knownA, 2, ctrl, cfg, mode);

angularAcc = max(-cfg.max_euler_acceleration(:), min(cfg.max_euler_acceleration(:), angularAcc));
torque = cfg.inertia(:).*(emap\angularAcc);
end


function ctrl = observe(ctrl, cfg, desiredA, measuredA)
attError = quad_math('wrap', desiredA-measuredA);
ctrl.attIntegral = max(-0.2, min(0.2, ctrl.attIntegral+cfg.dt*attError));
rateError = ctrl.nl{2}.v{2}-ctrl.nl{2}.z{2};
ctrl.obs = [attError/0.05;ctrl.attIntegral/0.05;rateError/0.5;measuredA/0.5];

if ctrl.td3.observe_gains
    limits = ctrl.td3.action_range;
    ctrl.obs = [ctrl.obs;2*(ctrl.attWc-limits(1))/(limits(2)-limits(1))-1];
end
end


function [target, ctrl] = allocate(ctrl, cfg, force, rmat, torque)
requested = ctrl.invMixer*[max(0, dot(force, rmat(:,3)));torque];
maximum = cfg.max_thrust_to_weight*cfg.mass*cfg.gravity/4;
target = max(0, min(maximum, requested));
ctrl.saturated = any(abs(target-requested) > 1e-9);
end


function state = rk4_step(t, state, target, cfg, mixer)
% one fourth-order Runge-Kutta step; k1 to k4 are the four slope estimates
dt = cfg.dt;
f = @(tt, xx) plant_rhs(tt, xx, target, cfg, mixer);

k1 = f(t, state);
k2 = f(t+dt/2, state+dt*k1/2);
k3 = f(t+dt/2, state+dt*k2/2);
k4 = f(t+dt, state+dt*k3);

state = state+dt*(k1+2*k2+2*k3+k4)/6;
state(7:10) = state(7:10)/norm(state(7:10));
end


function m = flight_metrics(log, dt)
err = log.reference-log.state(:,1:3);
ae = quad_math('wrap', log.attitude_reference-log.euler);

m.position_rmse_m = sqrt(mean(sum(err.^2, 2)));
m.peak_position_error_m = max(vecnorm(err, 2, 2));
m.attitude_rmse_deg = rad2deg(sqrt(mean(sum(ae.^2, 2))));
m.rotor_thrust_variation_N = sum(abs(diff(log.state(:,14:17))), 'all');
m.rotor_effort_N2s = dt*sum(log.state(:,14:17).^2, 'all');
m.saturated_sample_percent = 100*mean(log.saturated);
end


function bank = observer_bank(bandwidth, levels, ratio, dt)
% Linear ESO (levels = 1) or cascade ESO (levels = 2) for the three axes of one loop.
% bandwidth: observer bandwidth w0 per axis (rad/s)
% ratio: spacing between cascade stages, so stage bandwidths are w0/ratio and w0
% x: observer states, one column per axis; each stage has 3 rows:
%    [tracking-error estimate; error-rate estimate; disturbance estimate]
bank.x = zeros(3*levels, 3);
bank.initialized = false;
bank.Ad = cell(1, 3);
bank.Bd = cell(1, 3);

for channel = 1:3
    % A, B: continuous-time observer matrices for this axis
    n = 3*levels;
    A = zeros(n);
    B = zeros(n, 2);

    for i = 1:levels
        % w: this stage's bandwidth; gain: its observer gains [3w; 3w^2; w^3]
        w = bandwidth(channel)*ratio^(i-levels);
        gain = [3*w;3*w*w;w^3];
        j = 3*(i-1)+1;

        A(j:j+2,j:j+2) = [-gain(1) 1 0;-gain(2) 0 1;-gain(3) 0 0];
        B(j+1,2) = -1;

        if i == 1
            % the first stage is driven by the measured error
            B(1:3,1) = gain;
        else
            % later stages track the previous stage and add its disturbance estimate
            A(j:j+2,j-3) = A(j:j+2,j-3)+gain;
            for previous = 1:i-1
                A(j+1,3*previous) = A(j+1,3*previous)+1;
            end
        end
    end

    % exact discretisation: [Ad Bd] = top rows of expm([A B; 0 0] * dt)
    D = expm([A B;zeros(2,n+2)]*dt);
    bank.Ad{channel} = D(1:n,1:n);
    bank.Bd{channel} = D(1:n,n+1:n+2);
end
end


function [u, fstar, ctrl] = feedback(desired, measured, known, index, ctrl, cfg, mode)
if ismember(mode, {'tanh', 'td3'})
    [u, fstar, ctrl.nl{index}] = tanh_adrc_update(ctrl.nl{index}, desired, measured, known);
    return
end

if index == 1
    err = desired-measured;
    wc = cfg.position_bandwidth(:);
else
    err = quad_math('wrap', desired-measured);
    wc = cfg.attitude_bandwidth(:);
end

if strcmp(mode, 'pid')
    [u, ctrl] = pid_feedback(ctrl, cfg, err, wc, index);
    fstar = zeros(3, 1);
    return
end

bank = ctrl.banks{index};
if ~bank.initialized
    bank.x(1:3:end,:) = repmat(err', size(bank.x,1)/3, 1);
    bank.initialized = true;
end

for channel = 1:3
    bank.x(:,channel) = bank.Ad{channel}*bank.x(:,channel)+bank.Bd{channel}*[err(channel);known(channel)];
end

z1 = bank.x(end-2,:)';
z2 = bank.x(end-1,:)';
fstar = sum(bank.x(3:3:end,:), 1)';
u = wc.^2.*z1+2*wc.*z2+fstar;
ctrl.banks{index} = bank;
end


function [u, ctrl] = pid_feedback(ctrl, cfg, err, wc, index)
cutoff = [12 35];
bound = [2 0.3];
ki = [cfg.pid_position_integral_gain cfg.pid_attitude_integral_gain];

if ~ctrl.initialized(index)
    ctrl.lastError(index,:) = err';
    ctrl.initialized(index) = true;
end

delta = err-ctrl.lastError(index,:)';
if index == 2
    delta = quad_math('wrap', delta);
end

alpha = exp(-cutoff(index)*cfg.dt);
ctrl.derivative(index,:) = alpha*ctrl.derivative(index,:)+(1-alpha)*delta'/cfg.dt;

if ~ctrl.saturated
    ctrl.integral(index,:) = max(-bound(index), min(bound(index), ctrl.integral(index,:)+cfg.dt*err'));
end

ctrl.lastError(index,:) = err';
u = wc.^2.*err+2*wc.*ctrl.derivative(index,:)'+ki(index)*ctrl.integral(index,:)';
end


function st = tanh_adrc_init(cfg, prefix, angular)
% Nonlinear ADRC from Cai et al.: tracking differentiator, tanh-ESO and tanh feedback law.
% k: slope of each tanh term at zero, alpha * log(1/delta)
% beta: ESO gains {b0, b1, b2}; beta01, beta02: feedback gains set by bandwidth wc
% speed: tracking-differentiator speed r; lam: its gains [l1, l2]
% v: {v1, v2}, the smoothed reference and its rate
% z: {z1, z2, z3}, estimates of the output, its rate and the total disturbance
st.dt = cfg.dt;
st.angular = angular;
st.delta = cfg.tanh_delta;
st.alpha = cfg.tanh_alpha(:)';

k = st.alpha*log(1/st.delta);
wc = cfg.([prefix '_bandwidth'])(:);
wo = cfg.([prefix '_observer_bandwidth'])(:);
st.k = k;
st.beta = {3*wo, 3*wo.^2/k(1), wo.^3/k(1)};
st = tanh_set_bandwidth(st, wc);

st.speed = cfg.([prefix '_td_speed']);
st.lam = [1/k(1), 2/k(1)];
st.feedforward = logical(cfg.tanh_acceleration_feedforward);
st.v = [];
st.z = [];
end


function st = tanh_set_bandwidth(st, wc)
% wc: control bandwidth (rad/s); the TD3 agent changes it during flight
st.beta01 = wc.^2/st.k(2);
st.beta02 = 2*wc/st.k(3);
end


function ctrl = set_td3_action(ctrl, action)
limits = ctrl.td3.action_range;
low = limits(1);
high = limits(2);
ctrl.wcTarget = low+(max(-1, min(1, action))+1)*(high-low)/2;

if ctrl.maxStep <= 0
    ctrl.attWc = ctrl.wcTarget;
    ctrl.nl{2} = tanh_set_bandwidth(ctrl.nl{2}, ctrl.attWc);
end
end


function x = actor_forward(layers, x)
for j = 1:numel(layers)
    x = layers(j).W*x+layers(j).b;
    if j == numel(layers)
        x = tanh(x);
    else
        x = max(x, 0);
    end
end
end


function [u, fstar, st] = tanh_adrc_update(st, desired, measured, known)
dt = st.dt;
d = st.delta;
a = st.alpha;

if isempty(st.v)
    st.v = {desired(:), zeros(3,1)};
    st.z = {measured(:), zeros(3,1), zeros(3,1)};
end

% tracking differentiator; vdot is the smoothed reference acceleration
v1 = st.v{1};
v2 = st.v{2};
r = st.speed;
vdot = r*r*(-st.lam(1)*tanh_fn(tanh_diff(st, v1, desired), a(1), d)-st.lam(2)*tanh_fn(v2/r, a(1), d));
v2 = v2+dt*vdot;
v1 = v1+dt*v2;

% tanh-ESO; e is the estimation error and fe its tanh
z1 = st.z{1};
z2 = st.z{2};
z3 = st.z{3};
e = tanh_diff(st, z1, measured);
fe = tanh_fn(e, a(1), d);
z1n = z1+dt*(z2-st.beta{1}.*e);
z2n = z2+dt*(z3-st.beta{2}.*fe+known);
z3n = z3+dt*(-st.beta{3}.*fe);

if st.angular
    v1 = quad_math('wrap', v1);
    z1n = quad_math('wrap', z1n);
end
st.v = {v1, v2};
st.z = {z1n, z2n, z3n};

% feedback law u0, then cancel the estimated disturbance z3
u0 = st.beta01.*tanh_fn(tanh_diff(st, v1, z1n), a(2), d)+st.beta02.*tanh_fn(v2-z2n, a(3), d);
if st.feedforward
    u0 = u0+vdot;
end

u = u0-z3n;
fstar = vdot-z3n;
end


function out = tanh_fn(e, alpha, delta)
% the paper's tanh function: tanh(alpha * log(1/delta) * e)
out = tanh(alpha*log(1/delta)*e);
end


function out = tanh_diff(st, a, b)
% angle differences are wrapped to [-pi, pi)
if st.angular
    out = quad_math('wrap', a-b);
else
    out = a-b;
end
end


function out = plant_rhs(t, s, target, cfg, mixer)
% time derivative of the 17-value state s:
%   s(1:3) position, s(4:6) velocity, s(7:10) quaternion [w x y z],
%   s(11:13) body rates [p q r], s(14:17) actual motor thrusts
% target: motor thrusts the controller asked for (N)
% mass, J: real mass and inertia (different from the controller's if mismatch is on)
mass = cfg.mass;
J = cfg.inertia(:);
if cfg.enable_mismatch
    mass = mass*cfg.mass_mismatch_scale;
    J = J.*cfg.inertia_mismatch_scale(:);
end

v = s(4:6);
q = s(7:10);
omega = s(11:13);
thrusts = s(14:17);

% wrench: [total thrust; three body torques]; force, torque: wind gusts
R = quad_math('rotation', q);
wrench = mixer*thrusts;
[force, torque] = quad_disturbance(t, cfg.enable_disturbance);

acc = (R(:,3)*wrench(1)+force-cfg.linear_drag*v)/mass-[0;0;cfg.gravity];
angular = (wrench(2:4)+torque-cfg.angular_drag*omega-cross(omega, J.*omega))./J;

% quaternion derivative from the body rates
w = q(1);
x = q(2);
y = q(3);
z = q(4);
wx = omega(1);
wy = omega(2);
wz = omega(3);
qdot = 0.5*[-x*wx-y*wy-z*wz;w*wx+y*wz-z*wy;w*wy+z*wx-x*wz;w*wz+x*wy-y*wx];

% each motor moves toward its target with a first-order lag
out = [v;acc;qdot;angular;(target-thrusts)/cfg.motor_time_constant];
end


function [state, n] = sample_noise(state)
u = zeros(6, 1);
for k = 1:6
    state = mod(16807*state, 2147483647);
    u(k) = state/2147483647;
end

r = sqrt(-2*log(u(1:2:end)));
phase = 2*pi*u(2:2:end);

n = zeros(6, 1);
n(1:2:end) = r.*cos(phase);
n(2:2:end) = r.*sin(phase);
end
