function [estimatedPos, RMSE] = runFilter(filterClass, data, config)
    % Shared parallel execution loop for KF/PF-like filters.
    % The filter class must implement:
    %   initializeState(numPoints)
    %   initializeFirstTwo(state, iterIdx)
    %   step(state, iterIdx, pointIdx)

    srcDir = fileparts(fileparts(mfilename('fullpath')));
    addpath(fullfile(srcDir, 'Filters'));

    numNoise = numel(config.noiseVariance);
    numPoints = size(data.x_hat_LLS, 2);
    numIterations = config.iterations;

    estimatedPos = zeros(2, numPoints, numIterations, numNoise);
    RMSE = zeros(numNoise, 1);

    parfor noiseIdx = 1:numNoise
        filterObj = localCreateFilter(filterClass, data, config, noiseIdx);
        estNoise = zeros(2, numPoints, numIterations);

        for iterIdx = 1:numIterations
            state = filterObj.initializeState(numPoints);
            [state, p1, p2] = filterObj.initializeFirstTwo(state, iterIdx);
            estNoise(:, 1, iterIdx) = p1;
            estNoise(:, 2, iterIdx) = p2;

            for pointIdx = 3:numPoints
                [state, est] = filterObj.step(state, iterIdx, pointIdx);
                estNoise(:, pointIdx, iterIdx) = est;
            end
        end

        estimatedPos(:, :, :, noiseIdx) = estNoise;
        [RMSE(noiseIdx), ~] = evaluateFilter(estNoise, 3);
    end
end

function filterObj = localCreateFilter(filterClass, data, config, noiseIdx)
    if isa(filterClass, 'function_handle')
        filterObj = filterClass(data, config, noiseIdx);
        return;
    end

    className = lower(strtrim(char(filterClass)));
    switch className
        case 'linearkalmanfilter'
            filterObj = LinearKalmanFilter(data, config, noiseIdx);
        case 'linearkalmanfilter_decayq'
            filterObj = LinearKalmanFilter_DecayQ(data, config, noiseIdx);
        case 'linearparticlefilter'
            filterObj = LinearParticleFilter(data, config, noiseIdx);
        case 'nonlinearparticlefilter'
            filterObj = NonlinearParticleFilter(data, config, noiseIdx);
        case 'customnonlinearparticlefilter'
            filterObj = CustomNonlinearParticleFilter(data, config, noiseIdx);
        case 'adaptiverparticlefilter'
            filterObj = AdaptiveRParticleFilter(data, config, noiseIdx);
        case 'baseline'
            filterObj = Baseline(data, config, noiseIdx);

        otherwise
            error('runFilter:UnsupportedFilter', 'Unsupported filterClass: %s', char(filterClass));
    end
end