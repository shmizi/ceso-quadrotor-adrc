function summary = validate_matlab()
root = fileparts(fileparts(mfilename('fullpath')));
cfg = jsondecode(fileread(fullfile(root, 'config.json')));
folder = fullfile(root, 'results');
if ~exist(folder, 'dir')
    mkdir(folder);
end

modes = {'pid', 'eso', 'ceso', 'tanh', 'td3'};
paths = {'hover', 'circle', 'figure8', 'waypoints'};
paper = cfg;
paper.td3_variant = 'paper';
cases = [modes {'td3'}];
configs = [repmat({cfg}, 1, numel(modes)) {paper}];
labels = [modes {'td3_paper'}];
rows = {};
defaults = cell(1, numel(modes));

for j = 1:numel(cases)
    for k = 1:4
        try
            data = quad_simulate(configs{j}, cases{j}, paths{k});
        catch err
            if ~strcmp(cases{j}, 'td3') || ~strcmp(err.identifier, 'quad:Envelope')
                rethrow(err);
            end
            rows(end+1,:) = {labels{j}, paths{k}, NaN, NaN, NaN, NaN, NaN};
            fprintf('FAIL %-9s %-10s %s\n', labels{j}, paths{k}, err.message);
            continue
        end

        m = data.metrics;
        rows(end+1,:) = {labels{j}, paths{k}, m.position_rmse_m, m.peak_position_error_m, ...
            m.attitude_rmse_deg, m.rotor_thrust_variation_N, m.saturated_sample_percent};
        fprintf('PASS %-9s %-10s RMSE %.5f m\n', labels{j}, paths{k}, m.position_rmse_m);

        if j > numel(modes)
            continue
        end
        assert(all(isfinite(data.state), 'all'), 'Nonfinite plant state.');
        assert(max(abs(vecnorm(data.state(:,7:10), 2, 2)-1)) < 1e-12, 'Quaternion normalization.');
        assert(max(abs(data.euler(:,1:2)), [], 'all') < deg2rad(45), 'Attitude envelope.');
        assert(m.position_rmse_m < 0.5, 'Tracking accuracy outside baseline envelope.');

        if strcmp(paths{k}, 'figure8')
            defaults{j} = data;
            quad_export(data, fullfile(folder, ['matlab_' modes{j} '_figure8.csv']));
        end
    end
end

clean = cfg;
clean.enable_noise = false;
clean.enable_disturbance = false;
clean.duration = 5;
for j = 1:numel(modes)
    data = quad_simulate(clean, modes{j}, 'hover');
    assert(max(abs(data.state(:,1:3)-[0 0 1]), [], 'all') < 1e-10, 'Equilibrium drift.');
end
fprintf('PASS undisturbed hover equilibrium (all five controllers)\n');

stress = cfg;
stress.enable_mismatch = true;
for path = {'circle', 'figure8', 'waypoints'}
    data = quad_simulate(stress, 'ceso', path{1});
    assert(data.metrics.position_rmse_m < 0.3, 'CESO mismatch tracking envelope.');
    fprintf('PASS CESO with mass/inertia mismatch, %s, RMSE %.5f m\n', path{1}, data.metrics.position_rmse_m);
end

fine = cfg;
fine.enable_noise = false;
fine.duration = 16;
coarse = quad_simulate(fine, 'ceso', 'figure8');
fine.dt = cfg.dt/2;
fineLog = quad_simulate(fine, 'ceso', 'figure8');
delta = max(abs(coarse.state(:,1:3)-fineLog.state(1:2:end,1:3)), [], 'all');
assert(delta < 0.025, 'Timestep convergence check.');
fprintf('PASS dt versus dt/2 maximum position discrepancy %.6f m\n', delta);

summary = cell2table(rows, 'VariableNames', {'controller', 'trajectory', 'position_rmse_m', ...
    'peak_position_error_m', 'attitude_rmse_deg', 'rotor_thrust_variation_N', 'saturation_percent'});
writetable(summary, fullfile(folder, 'matlab_validation.csv'));

[fig, ~] = run_quadcopter_demo('Visible', 'off', 'Animate', false);
quad_capture(fig, fullfile(folder, 'matlab_dashboard.png'));
close(fig);

fig = quad_compare_plot(defaults(~cellfun(@isempty, defaults)), 'off');
exportgraphics(fig, fullfile(folder, 'matlab_comparison.png'), 'Resolution', 140, 'BackgroundColor', fig.Color);
close(fig);

save(fullfile(folder, 'matlab_validated_logs.mat'), 'defaults', 'summary', 'delta', 'cfg');
fprintf('PASS dashboard and comparison render\n');
end
