function [estimatedPos, metric] = runFilter(filterClass, data, config)
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
    APE = zeros(numNoise, 1);

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
        [RMSE(noiseIdx), APE(noiseIdx)] = evaluateFilter(estNoise, 3);
    end
    metric.RMSE = RMSE;
    metric.APE = APE;
end

function filterObj = localCreateFilter(filterClass, data, config, noiseIdx)
    if isa(filterClass, 'function_handle')
        filterObj = filterClass(data, config, noiseIdx);
        return;
    end

    className = lower(strtrim(char(filterClass)));
    switch className
        case 'baseline'
            filterObj = Baseline(data, config, noiseIdx);
        case 'linearkalmanfilter_decayq'
            filterObj = LinearKalmanFilter_DecayQ(data, config, noiseIdx);
        case 'nonlinearparticlefilter'
            filterObj = NonlinearParticleFilter(data, config, noiseIdx);
        case 'customnonlinearparticlefilter'
            filterObj = CustomNonlinearParticleFilter(data, config, noiseIdx);
        case 'ekfparticlefilter'
            filterObj = EKFParticleFilter(data, config, noiseIdx);
        case 'adaptiveparticlefilter'
            [bestBeta, bestLambdaR] = getBestParams(noiseIdx);
            filterObj = AdaptiveParticleFilter(data, config, noiseIdx, bestBeta, bestLambdaR);
        case {'beliefqshrinkadaptiveparticlefilter', 'bqspf'}
            [bestBeta, bestLambdaR] = getBestParams(noiseIdx);
            filterObj = BeliefQShrinkAdaptiveParticleFilter(data, config, noiseIdx, bestBeta, bestLambdaR);
        case {'rdiagprioreditadaptiveparticlefilter', 'rdpepf'}
            [bestBeta, bestLambdaR] = getBestParams(noiseIdx);
            filterObj = RDiagPriorEditAdaptiveParticleFilter(data, config, noiseIdx, bestBeta, bestLambdaR);
        case {'beliefrougheningadaptiveparticlefilter', 'brapf'}
            [bestBeta, bestLambdaR] = getBestParams(noiseIdx);
            filterObj = BeliefRougheningAdaptiveParticleFilter(data, config, noiseIdx, bestBeta, bestLambdaR);

        otherwise
            error('runFilter:UnsupportedFilter', 'Unsupported filterClass: %s', char(filterClass));
    end
end