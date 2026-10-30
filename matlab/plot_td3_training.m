function f = plot_td3_training(csvFile)
if nargin < 1
    root = fileparts(fileparts(mfilename('fullpath')));
    cfg = jsondecode(fileread(fullfile(root, 'config.json')));
    csvFile = fullfile(root, cfg.(['td3_' cfg.td3_variant]).training_log);
end

T = readtable(csvFile, 'VariableNamingRule', 'preserve');
f = figure('Name', 'TD3 training', 'NumberTitle', 'off', 'Position', [80 80 1100 700]);
tl = tiledlayout(f, 2, 1, 'Padding', 'compact');

ax = nexttile(tl);
hold(ax, 'on');
grid(ax, 'on');
plot(ax, T.episode, T.("return"), 'Color', [.7 .7 .7]);
plot(ax, T.episode, movmean(T.("return"), 20), 'b', 'LineWidth', 1.5);
evaluated = ~isnan(T.evaluation_return);
plot(ax, T.episode(evaluated), T.evaluation_return(evaluated), 'ro-');
ylabel(ax, 'Return');
title(ax, 'Episode return');
legend(ax, {'Training episode', '20-episode mean', 'Evaluation'}, 'Location', 'southeast');

ax = nexttile(tl);
plot(ax, T.episode, [T.mean_wc_roll, T.mean_wc_pitch, T.mean_wc_yaw]);
grid(ax, 'on');
xlabel(ax, 'Episode');
ylabel(ax, '\omega_c (rad/s)');
title(ax, 'Mean commanded attitude bandwidth');
legend(ax, {'Roll', 'Pitch', 'Yaw'});
end
