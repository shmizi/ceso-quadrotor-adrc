function [force, torque] = quad_disturbance(t, enabled)
if ~enabled
    force = zeros(3, 1);
    torque = zeros(3, 1);
    return
end

a = pulse(t, 9, 13);
b = pulse(t, 18, 23);
force = a*[3;-2;1.2]+b*[-2;2.5;0.6*sin(1.3*t)];
torque = a*[0.04;-0.03;0.02]+b*[0.03*sin(t);0.03;-0.02];
end


function out = pulse(t, start, finish)
s = max(0, min(1, (t-start)/0.5));
f = max(0, min(1, (finish-t)/0.5));
out = (3*s*s-2*s^3)*(3*f*f-2*f^3);
end
