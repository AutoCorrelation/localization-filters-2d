classdef DataGenerator
    % DataGenerator generates synthetic trajectory data and corresponding noisy Time of Arrival (ToA) measurements for localization tasks.
    % It supports multiple trajectory types (constant velocity, circular, zigzag) and allows for configurable noise levels.
    % The generated data is saved in HDF5 format for use in localization filter evaluations.
    properties
        gt_pos
        trajectory
        anchors = [0 10; 0 0; 10 0; 10 10]'
    end

    methods
        function obj = DataGenerator(trajectory, steps)
            if nargin < 1
                trajectory = "cv";
                steps = 11;
            elseif nargin < 2       
                steps = 11;
            end
            if trajectory == "cv"
                x = linspace(0, 10, steps);
                y = linspace(0, 10, steps);
                obj.gt_pos = [x; y];
                obj.trajectory = trajectory;

            elseif trajectory == "circular"
                center = [5; 5];
                R = 3;
                x = linspace(0, 2 * pi, steps);
                y = linspace(0, 2 * pi, steps);
                obj.gt_pos = center + R * [cos(x + pi); sin(y + pi)];
                obj.trajectory = trajectory;
            elseif trajectory == "zigzag"
                x = linspace(0, 10, steps);
                y = 5 + 3 * sin(4 * pi * x / 10);
                obj.gt_pos = [x; y];
                obj.trajectory = trajectory;
            else
                error('Unsupported trajectory type. Use "cv", "circular", or "zigzag".');
            end


            noisyToA(obj);
            visualize(obj);
        end

        function visualize(obj)
            figure;
            % time = 0:size(obj.gt_pos, 2) - 1;
            plot(obj.gt_pos(1, :), obj.gt_pos(2, :), 'b-', 'LineWidth', 2, 'Marker', 'o', 'MarkerSize', 8, 'MarkerFaceColor', 'r');
            xlabel('X');
            ylabel('Y');
            title(['Trajectory: ', obj.trajectory]);
            hold on;
            axis equal;
            grid on;
            plot(obj.anchors(1, :), obj.anchors(2, :), 'r^', 'MarkerSize', 10, 'LineWidth', 2);
            legend('True State', 'Anchors');

        end

        function noisyToA(obj, h5Path)
            numData = 11000;
            if nargin < 2 || isempty(h5Path)
                dataDir = fileparts(mfilename('fullpath'));
                h5Path = fullfile(dataDir, char("ranging_data_" + string(obj.trajectory) + ".h5"));
            end

            gt_ranging = sqrt((obj.gt_pos(1, :) - obj.anchors(1, :)').^2 + (obj.gt_pos(2, :) - obj.anchors(2, :)').^2); % [anchor, step]
            noiseVar = [0.01, 0.1, 1, 10, 100];
            shape = [size(gt_ranging), numData]; % [anchor, step, sample]
            noise_001 = sqrt(noiseVar(1)) * randn(shape);
            noise_01 = sqrt(noiseVar(2)) * randn(shape);
            noise_1 = sqrt(noiseVar(3)) * randn(shape);
            noise_10 = sqrt(noiseVar(4)) * randn(shape);
            noise_100 = sqrt(noiseVar(5)) * randn(shape);

            ranging_001 = gt_ranging + noise_001;
            ranging_01 = gt_ranging + noise_01;
            ranging_1 = gt_ranging + noise_1;
            ranging_10 = gt_ranging + noise_10;
            ranging_100 = gt_ranging + noise_100;

            if exist(h5Path, 'file') == 2
                delete(h5Path);
            end

            h5create(h5Path, '/gt_position', size(obj.gt_pos)); % [2, step]
            h5write(h5Path, '/gt_position', obj.gt_pos);

            h5create(h5Path, '/gt_ranging', size(gt_ranging)); % [anchor, step]
            h5write(h5Path, '/gt_ranging', gt_ranging);

            h5create(h5Path, '/ranging_001', size(ranging_001));
            h5write(h5Path, '/ranging_001', ranging_001);

            h5create(h5Path, '/ranging_01', size(ranging_01));
            h5write(h5Path, '/ranging_01', ranging_01);

            h5create(h5Path, '/ranging_1', size(ranging_1));
            h5write(h5Path, '/ranging_1', ranging_1);

            h5create(h5Path, '/ranging_10', size(ranging_10));
            h5write(h5Path, '/ranging_10', ranging_10);

            h5create(h5Path, '/ranging_100', size(ranging_100));
            h5write(h5Path, '/ranging_100', ranging_100);

            allRanging = cat(4, ranging_001, ranging_01, ranging_1, ranging_10, ranging_100); % five x [anchor,step,sample] -> [anchor,step,sample,noise]
            h5create(h5Path, '/allRanging', size(allRanging));
            h5write(h5Path, '/allRanging', allRanging);
        end
    end
end 