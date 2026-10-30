function f = quad_compare_plot(logs, visible)
if nargin < 2
    visible = 'on';
end

names = containers.Map({'pid', 'eso', 'ceso', 'tanh', 'td3'}, {'PID', 'ESO', 'CESO', 'tanh-ADRC', 'TD3-ADRC'});
labels = cellfun(@(d) names(d.mode), logs, 'UniformOutput', false);

f = figure('Name', 'Controller comparison', 'NumberTitle', 'off', 'Position', [90 90 1300 780], 'Visible', visible);
tl = tiledlayout(f, 2, 3, 'Padding', 'compact', 'TileSpacing', 'compact');

ax = nexttile(tl, [1 2]);
hold(ax, 'on');
grid(ax, 'on');
for j = 1:numel(logs)
    plot(ax, logs{j}.time, vecnorm(logs{j}.reference-logs{j}.state(:,1:3), 2, 2));
end
xlabel(ax, 'Time (s)');
ylabel(ax, 'Position error (m)');
legend(ax, labels);
title(ax, 'Position error');

ax = nexttile(tl);
hold(ax, 'on');
grid(ax, 'on');
for j = 1:numel(logs)
    plot3(ax, logs{j}.state(:,1), logs{j}.state(:,2), logs{j}.state(:,3));
end
plot3(ax, logs{1}.reference(:,1), logs{1}.reference(:,2), logs{1}.reference(:,3), 'k--');
view(ax, [-35 25]);
xlabel(ax, 'x (m)');
ylabel(ax, 'y (m)');
zlabel(ax, 'z (m)');
title(ax, 'Trajectory');

fields = {'position_rmse_m', 'attitude_rmse_deg', 'rotor_thrust_variation_N'};
titles = {'Position RMSE (m)', 'Attitude RMSE (deg)', 'Rotor thrust variation (N)'};
for metric = 1:3
    ax = nexttile(tl);
    bar(ax, categorical(labels, labels), cellfun(@(d) d.metrics.(fields{metric}), logs));
    grid(ax, 'on');
    title(ax, titles{metric});
end
end
