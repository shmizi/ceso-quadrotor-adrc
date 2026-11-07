function f = plot_sensitivity(csvFile, visible)
if nargin < 1 || isempty(csvFile)
    csvFile = fullfile(fileparts(fileparts(mfilename('fullpath'))), 'results', 'sensitivity.csv');
end
if nargin < 2
    visible = 'on';
end

T = readtable(csvFile, 'VariableNamingRule', 'preserve', 'TextType', 'string', 'Delimiter', ',');
T.setting = str2double(string(T.setting));

% study, legend name, colour, marker, and which settings to label on each line
studies = {'attitude_w0', 'Attitude \omega_0', [42 120 214], 'o', [12 100]; ...
           'attitude_wc', 'Attitude \omega_c', [235 104 52], 's', [1 28]; ...
           'position_w0', 'Position \omega_0', [27 175 122], '^', [4 14 28]; ...
           'position_wc', 'Position \omega_c', [237 161 0], 'd', [0.5 3]; ...
           'cascade_ratio', 'Cascade ratio', [232 123 164], 'v', [1.25 4]};

f = figure('Name', 'CESO sensitivity', 'NumberTitle', 'off', 'Position', [80 80 1300 620], ...
    'Color', 'w', 'Visible', visible);
theme(f, 'light');
tl = tiledlayout(f, 1, 2, 'Padding', 'compact', 'TileSpacing', 'compact');

fields = {'position_rmse_m', 'attitude_rmse_deg'};
labels = {'Position RMSE (m)', 'Attitude RMSE (deg)'};
titles = {'Position error against motor activity', 'Attitude error against motor activity'};
yticks_ = {[0.02 0.05 0.1 0.2 0.3], [0.2 0.5 1 2 5 10]};
ylimits = {[0.018 0.33], [0.13 12]};

for panel = 1:2
    ax = nexttile(tl);
    hold(ax, 'on');
    grid(ax, 'on');
    set(ax, 'XScale', 'log', 'YScale', 'log', 'GridColor', [.85 .85 .85], 'GridAlpha', 1, ...
        'MinorGridLineStyle', 'none', 'FontSize', 11, 'XColor', [.32 .32 .31], 'YColor', [.32 .32 .31]);

    lines = gobjects(size(studies, 1), 1);
    for j = 1:size(studies, 1)
        S = sortrows(T(T.study == studies{j,1}, :), 'setting');
        x = S.rotor_thrust_variation_N;
        y = S.(fields{panel});
        lines(j) = plot(ax, x, y, '-', 'Color', studies{j,3}/255, 'LineWidth', 2, ...
            'Marker', studies{j,4}, 'MarkerSize', 8, 'MarkerFaceColor', studies{j,3}/255, 'MarkerEdgeColor', 'w');

        for value = studies{j,5}
            k = find(S.setting == value, 1);
            if strcmp(studies{j,1}, 'cascade_ratio')
                text(ax, x(k), y(k), sprintf('%g  ', value), 'FontSize', 10, 'Color', [.32 .32 .31], ...
                    'HorizontalAlignment', 'right');
            elseif strcmp(studies{j,1}, 'position_w0')
                text(ax, x(k), y(k), sprintf('   %g', value), 'FontSize', 10, 'Color', [.32 .32 .31], ...
                    'VerticalAlignment', 'top');
            elseif strcmp(studies{j,1}, 'attitude_wc')
                text(ax, x(k), y(k), sprintf('%g', value), 'FontSize', 10, 'Color', [.32 .32 .31], ...
                    'VerticalAlignment', 'bottom');
            else
                text(ax, x(k), y(k), sprintf('  %g', value), 'FontSize', 10, 'Color', [.32 .32 .31]);
            end
        end
    end

    nominal = T(T.study == "position_w0" & T.setting == 10, :);
    star = plot(ax, nominal.rotor_thrust_variation_N, nominal.(fields{panel}), 'p', 'MarkerSize', 17, ...
        'MarkerFaceColor', [.04 .04 .04], 'MarkerEdgeColor', 'w');

    set(ax, 'XTick', [200 500 1000 2000 5000], 'YTick', yticks_{panel});
    xlim(ax, [150 6000]);
    ylim(ax, ylimits{panel});
    xlabel(ax, 'Rotor thrust variation (N)');
    ylabel(ax, labels{panel});
    title(ax, titles{panel}, 'FontWeight', 'normal', 'Color', [.04 .04 .04]);

    if panel == 1
        lgd = legend(ax, [lines;star], [studies(:,2);{'config.json'}], 'Orientation', 'horizontal', 'Box', 'off', 'FontSize', 11);
        lgd.Layout.Tile = 'south';
    end
end
end
