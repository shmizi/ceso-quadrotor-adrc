function [fig, lastLog] = run_quadcopter_demo(varargin)
p = inputParser;
addParameter(p, 'Animate', true);
addParameter(p, 'Visible', 'on');
addParameter(p, 'AutoRun', true);
parse(p, varargin{:});
opts = p.Results;

root = fileparts(fileparts(mfilename('fullpath')));
base = jsondecode(fileread(fullfile(root, 'config.json')));

lastLog = [];
comparison = {};
busy = false;
stopped = false;
paused = false;
h = struct();

trajectoryKeys = {'figure8', 'circle', 'waypoints', 'hover'};
modeKeys = {'ceso', 'eso', 'pid', 'tanh', 'td3'};
modeNames = {'CESO', 'ESO', 'PID', 'tanh-ADRC', 'TD3-ADRC'};

fig = figure('Name', 'Quadrotor simulation', 'NumberTitle', 'off', ...
    'Position', [60 60 1400 860], 'Visible', opts.Visible);
controls = uipanel(fig, 'Title', 'Simulation', 'Units', 'normalized', 'Position', [.01 .80 .98 .13]);

text_label(controls, [.01 .55 .07 .3], 'Trajectory');
trajectory = uicontrol(controls, 'Style', 'popupmenu', 'Units', 'normalized', 'Position', [.08 .56 .12 .32], ...
    'String', {'Figure eight', 'Circle', 'Waypoints', 'Hover'});
text_label(controls, [.22 .55 .07 .3], 'Controller');
controller = uicontrol(controls, 'Style', 'popupmenu', 'Units', 'normalized', 'Position', [.29 .56 .12 .32], ...
    'String', modeNames);
trajectory.Value = find(strcmp(trajectoryKeys, base.default_trajectory), 1);
controller.Value = find(strcmp(modeKeys, base.default_controller), 1);

gust = checkbox(controls, [.44 .56 .08 .3], 'Gusts', base.enable_disturbance);
noise = checkbox(controls, [.52 .56 .07 .3], 'Noise', base.enable_noise);
mismatch = checkbox(controls, [.59 .56 .11 .3], 'Model mismatch', base.enable_mismatch);

runButton = button(controls, [.72 .55 .08 .36], 'Run', @run_selected);
compareButton = button(controls, [.81 .55 .09 .36], 'Compare all', @compare_all);
text_label(controls, [.91 .55 .04 .3], 'Speed');
speed = uicontrol(controls, 'Style', 'popupmenu', 'Units', 'normalized', 'Position', [.945 .56 .05 .32], ...
    'String', {'1x', '2x', '4x', '8x'}, 'Value', 2);
pauseButton = uicontrol(controls, 'Style', 'togglebutton', 'Units', 'normalized', 'Position', [.72 .08 .08 .36], ...
    'String', 'Pause', 'Callback', @pause_toggle);
button(controls, [.81 .08 .055 .36], 'Stop', @stop_run);
button(controls, [.87 .08 .06 .36], 'Observers', @show_observers);
button(controls, [.935 .08 .06 .36], 'Export', @save_log);
status = text_label(controls, [.01 .08 .69 .3], 'Ready.');

axFlight = axes(fig, 'Position', [.05 .33 .42 .40]);
axPosition = axes(fig, 'Position', [.56 .58 .41 .15]);
axAttitude = axes(fig, 'Position', [.56 .34 .41 .15]);
axMotor = axes(fig, 'Position', [.56 .07 .41 .19]);
axForce = axes(fig, 'Position', [.05 .07 .42 .16]);
set([axFlight axPosition axAttitude axMotor axForce], 'NextPlot', 'add', ...
    'XGrid', 'on', 'YGrid', 'on', 'ZGrid', 'on', 'Box', 'on');

if opts.AutoRun
    run_selected();
end


    function obj = text_label(parent, pos, str)
        obj = uicontrol(parent, 'Style', 'text', 'Units', 'normalized', 'Position', pos, ...
            'String', str, 'HorizontalAlignment', 'left');
    end


    function obj = checkbox(parent, pos, str, value)
        obj = uicontrol(parent, 'Style', 'checkbox', 'Units', 'normalized', 'Position', pos, ...
            'String', str, 'Value', value);
    end


    function obj = button(parent, pos, str, callback)
        obj = uicontrol(parent, 'Style', 'pushbutton', 'Units', 'normalized', 'Position', pos, ...
            'String', str, 'Callback', callback);
    end


    function [cfg, mode, name] = settings()
        cfg = base;
        cfg.enable_disturbance = logical(gust.Value);
        cfg.enable_noise = logical(noise.Value);
        cfg.enable_mismatch = logical(mismatch.Value);
        name = trajectoryKeys{trajectory.Value};
        mode = modeKeys{controller.Value};
    end


    function toggle_busy(value)
        busy = value;
        if ~isgraphics(fig)
            return
        end

        if value
            state = 'off';
        else
            state = 'on';
        end
        set([runButton compareButton trajectory controller gust noise mismatch], 'Enable', state);
    end


    function report(frac)
        if ~isgraphics(fig) || stopped
            error('quad:Cancelled', 'Simulation stopped.');
        end
        status.String = sprintf('Running %s: %3.0f%%', modeNames{controller.Value}, 100*frac);
        drawnow;
    end


    function run_selected(varargin)
        if busy
            return
        end

        stopped = false;
        paused = false;
        pauseButton.Value = 0;
        pauseButton.String = 'Pause';
        toggle_busy(true);
        cleaner = onCleanup(@() toggle_busy(false));
        [cfg, mode, name] = settings();

        try
            comparison = {};
            lastLog = quad_simulate(cfg, mode, name, @report);
            build_scene(lastLog);

            if opts.Animate && strcmp(opts.Visible, 'on')
                k = 1;
                clockStart = tic;
                while k <= numel(lastLog.time)
                    if ~isgraphics(fig) || stopped
                        break
                    end
                    while paused && isgraphics(fig) && ~stopped
                        pause(.05);
                        drawnow;
                    end
                    if ~isgraphics(fig) || stopped
                        break
                    end

                    render_frame(lastLog, k);
                    drawnow;
                    pause(max(.001, 1/30-toc(clockStart)));
                    clockStart = tic;
                    k = k+max(1, round(2^(speed.Value-1)/(30*cfg.dt)));
                end
            end

            if isgraphics(fig)
                render_frame(lastLog, numel(lastLog.time));
                if stopped
                    status.String = ['Stopped.   ' status.String];
                end
            end
        catch err
            handle_error(err);
        end
    end


    function build_scene(data)
        arrayfun(@cla, [axFlight axPosition axAttitude axMotor axForce]);

        plot3(axFlight, data.reference(:,1), data.reference(:,2), data.reference(:,3), '--', 'Color', [.6 .6 .6]);
        h.trail = plot3(axFlight, nan, nan, nan, 'b', 'LineWidth', 1.5);
        h.target = plot3(axFlight, nan, nan, nan, 'ro', 'MarkerSize', 6);
        h.arms = gobjects(4, 1);
        h.rotors = gobjects(4, 1);
        for j = 1:4
            h.arms(j) = plot3(axFlight, nan, nan, nan, 'k', 'LineWidth', 2);
            h.rotors(j) = plot3(axFlight, nan, nan, nan, 'k');
        end
        h.nose = plot3(axFlight, nan, nan, nan, 'r', 'LineWidth', 2);
        h.wind = quiver3(axFlight, 0, 0, 0, 0, 0, 0, 0, 'm', 'LineWidth', 1.2);

        pmin = min(data.reference, [], 1)-.8;
        pmax = max(data.reference, [], 1)+.8;
        xlim(axFlight, [pmin(1) pmax(1)]);
        ylim(axFlight, [pmin(2) pmax(2)]);
        zlim(axFlight, [0 max(2.4, pmax(3))]);
        daspect(axFlight, [1 1 1]);
        view(axFlight, [-38 25]);
        xlabel(axFlight, 'x (m)');
        ylabel(axFlight, 'y (m)');
        zlabel(axFlight, 'z (m)');
        title(axFlight, 'Flight path');
        legend(axFlight, [h.trail h.target h.wind], {'Actual', 'Reference', 'Disturbance force'}, 'Location', 'northwest');

        h.position = gobjects(3, 1);
        h.attitude = gobjects(3, 1);
        h.motors = gobjects(4, 1);
        order = get(groot, 'defaultAxesColorOrder');
        for j = 1:3
            plot(axPosition, data.time, data.reference(:,j), '--', 'Color', order(j,:));
            h.position(j) = plot(axPosition, nan, nan, 'Color', order(j,:));
            plot(axAttitude, data.time, rad2deg(data.attitude_reference(:,j)), '--', 'Color', order(j,:));
            h.attitude(j) = plot(axAttitude, nan, nan, 'Color', order(j,:));
            plot(axForce, data.time, data.force(:,j), 'Color', order(j,:));
        end
        for j = 1:4
            h.motors(j) = plot(axMotor, nan, nan, 'Color', order(j,:));
        end
        h.cursors = [xline(axPosition, 0, 'k:') xline(axAttitude, 0, 'k:') xline(axMotor, 0, 'k:') xline(axForce, 0, 'k:')];

        set([axPosition axAttitude axMotor axForce], 'XLim', [0 data.time(end)]);
        positions = [data.state(:,1:3);data.reference];
        ylim(axPosition, [min(positions, [], 'all')-.15, max(positions, [], 'all')+.15]);
        maxAtt = max(5, max(abs(rad2deg([data.euler;data.attitude_reference])), [], 'all')*1.12);
        ylim(axAttitude, [-maxAtt maxAtt]);
        ylim(axMotor, [0, base.max_thrust_to_weight*base.mass*base.gravity/4*1.05]);
        ylim(axForce, [-3.6 3.6]);

        title(axPosition, 'Position (dashed: reference)');
        ylabel(axPosition, 'm');
        title(axAttitude, 'Attitude (dashed: command)');
        ylabel(axAttitude, 'deg');
        title(axMotor, 'Rotor thrust');
        ylabel(axMotor, 'N');
        xlabel(axMotor, 'Time (s)');
        title(axForce, 'Disturbance force');
        ylabel(axForce, 'N');
        xlabel(axForce, 'Time (s)');

        legend(axPosition, h.position, {'x', 'y', 'z'}, 'Orientation', 'horizontal', 'Location', 'northwest');
        legend(axAttitude, h.attitude, {'Roll', 'Pitch', 'Yaw'}, 'Orientation', 'horizontal', 'Location', 'northwest');
        legend(axMotor, h.motors, {'1', '2', '3', '4'}, 'Orientation', 'horizontal', 'Location', 'southwest');
        legend(axForce, {'x', 'y', 'z'}, 'Orientation', 'horizontal', 'Location', 'northwest');
        drawnow;
    end


    function render_frame(data, k)
        position = data.state(k,1:3)';
        R = quad_math('rotation', data.state(k,7:10));
        history = 1:5:k;
        if isempty(history) || history(end) ~= k
            history = [history k];
        end

        set(h.trail, 'XData', data.state(history,1), 'YData', data.state(history,2), 'ZData', data.state(history,3));
        set_xyz(h.target, data.reference(k,:)');

        a = base.arm_length/sqrt(2);
        rotorLocations = [a -a -a a;a a -a -a;0 0 0 0];
        circle = linspace(0, 2*pi, 25);
        for j = 1:4
            endpoint = position+R*rotorLocations(:,j);
            set_xyz(h.arms(j), [position endpoint]);
            set_xyz(h.rotors(j), position+R*(rotorLocations(:,j)+.08*[cos(circle);sin(circle);zeros(size(circle))]));
        end
        set_xyz(h.nose, [position position+R*[.2;0;0]]);

        force = data.force(k,:);
        set(h.wind, 'XData', position(1), 'YData', position(2), 'ZData', position(3), ...
            'UData', .17*force(1), 'VData', .17*force(2), 'WData', .17*force(3));

        for j = 1:3
            set(h.position(j), 'XData', data.time(history), 'YData', data.state(history,j));
            set(h.attitude(j), 'XData', data.time(history), 'YData', rad2deg(data.euler(history,j)));
        end
        for j = 1:4
            set(h.motors(j), 'XData', data.time(history), 'YData', data.state(history,13+j));
        end
        set(h.cursors, 'Value', data.time(k));

        err = vecnorm(data.reference(1:k,:)-data.state(1:k,1:3), 2, 2);
        status.String = sprintf('%s   t = %.2f s   RMSE = %.3f m   error = %.3f m   peak = %.3f m', ...
            modeNames{strcmp(modeKeys, data.mode)}, data.time(k), sqrt(mean(err.^2)), err(end), max(err));
    end


    function set_xyz(obj, points)
        set(obj, 'XData', points(1,:), 'YData', points(2,:), 'ZData', points(3,:));
    end


    function pause_toggle(varargin)
        paused = logical(pauseButton.Value);
        if paused
            pauseButton.String = 'Resume';
        else
            pauseButton.String = 'Pause';
        end
    end


    function stop_run(varargin)
        stopped = true;
        paused = false;
    end


    function compare_all(varargin)
        if busy
            return
        end

        stopped = false;
        toggle_busy(true);
        cleaner = onCleanup(@() toggle_busy(false));
        [cfg, ~, name] = settings();
        modes = {'pid', 'eso', 'ceso', 'tanh', 'td3'};

        try
            comparison = cell(1, numel(modes));
            for j = 1:numel(modes)
                status.String = sprintf('Comparing %d of %d: %s', j, numel(modes), modeNames{strcmp(modeKeys, modes{j})});
                drawnow;
                comparison{j} = quad_simulate(cfg, modes{j}, name, @comparison_progress);
            end

            lastLog = comparison{3};
            build_scene(lastLog);
            render_frame(lastLog, numel(lastLog.time));
            quad_compare_plot(comparison);
            status.String = 'Comparison done.';
        catch err
            handle_error(err);
        end
    end


    function comparison_progress(~)
        drawnow;
        if ~isgraphics(fig) || stopped
            error('quad:Cancelled', 'Comparison stopped.');
        end
    end


    function show_observers(varargin)
        if isempty(lastLog)
            return
        end

        f = figure('Name', 'Disturbance estimates', 'NumberTitle', 'off', 'Position', [120 100 1000 620]);
        rows = 2+strcmp(lastLog.mode, 'td3');
        layout = tiledlayout(f, rows, 1, 'Padding', 'compact');
        labels = {'Position f^* (m/s^2)', 'Attitude f^* (rad/s^2)'};

        for row = 1:2
            ax = nexttile(layout);
            plot(ax, lastLog.time, lastLog.estimate(:,3*row-2:3*row));
            grid(ax, 'on');
            ylabel(ax, labels{row});
            legend(ax, {'1', '2', '3'});
        end

        if rows == 3
            ax = nexttile(layout);
            stairs(ax, lastLog.time, lastLog.gains);
            grid(ax, 'on');
            ylabel(ax, '\omega_c (rad/s)');
            legend(ax, {'Roll', 'Pitch', 'Yaw'});
        end
        xlabel(ax, 'Time (s)');
    end


    function save_log(varargin)
        if isempty(lastLog)
            return
        end

        folder = fullfile(root, 'results');
        if ~exist(folder, 'dir')
            mkdir(folder);
        end

        stem = fullfile(folder, ['matlab_' lastLog.mode '_' lastLog.trajectory]);
        quad_export(lastLog, [stem '.csv']);
        data = lastLog;
        save([stem '.mat'], 'data', 'comparison');
        quad_capture(fig, [stem '.png']);

        for j = 1:numel(comparison)
            quad_export(comparison{j}, fullfile(folder, ['matlab_' comparison{j}.mode '_' lastLog.trajectory '.csv']));
        end
        status.String = ['Saved to ' folder];
    end


    function handle_error(err)
        if isgraphics(fig)
            status.String = err.message;
            if ~strcmp(err.identifier, 'quad:Cancelled')
                if strcmp(opts.Visible, 'on')
                    errordlg(err.message, 'Simulation error');
                end
                rethrow(err);
            end
        end
    end
end
