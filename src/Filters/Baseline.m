classdef Baseline
    properties
        xHatProjected
    end

    methods
        function obj = Baseline(data, config, noiseIdx)
            zNoise = squeeze(data.z_LLS(:, :, :, noiseIdx));
            numPoints = size(zNoise, 2);
            numIterations = size(zNoise, 3);

            obj.xHatProjected = zeros(2, numPoints, numIterations);
            for iterIdx = 1:numIterations
                obj.xHatProjected(:, :, iterIdx) = config.pinvH * zNoise(:, :, iterIdx);
            end
        end

        function state = initializeState(~, numPoints)
            state.estimatedPos = zeros(2, numPoints);
        end

        function [state, p1, p2] = initializeFirstTwo(obj, state, iterIdx)
            p1 = obj.xHatProjected(:, 1, iterIdx);
            p2 = obj.xHatProjected(:, 2, iterIdx);

            state.estimatedPos(:, 1) = p1;
            state.estimatedPos(:, 2) = p2;
        end

        function [state, est] = step(obj, state, iterIdx, pointIdx)
            est = obj.xHatProjected(:, pointIdx, iterIdx);
            state.estimatedPos(:, pointIdx) = est;
        end
    end
end
