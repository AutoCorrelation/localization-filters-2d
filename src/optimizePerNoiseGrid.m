clear
clc
format long

addpath('./utils');
addpath('./Filters');
initializeParpool(5);

% Base configuration
config = initializeConfig(1e3);
baseSeed = 42;
rng(baseSeed, 'twister');

% Tuning options (adjust as needed)
opt.splitRatios = [0.6, 0.2, 0.2];
opt.splitSeed = 42;
opt.tuneIterations = 200;     % iterations used in train/val grid search
opt.finalIterations = 500;    % iterations used for final test evaluation

% Grids
opt.betaGrid = [0.6, 0.8, 0.9, 0.98, 0.99];
opt.lambdaRGrid = [0.1, 0.5, 1, 2, 5, 10, 20, 50, 100];

h5File = fullfile(config.pathData, 'simulation_data.h5');
if ~isfile(h5File)
    dataGenerate(config);
end


targetFilters = {
    struct('name', 'AdaptiveParticleFilter', 'factory', @(d, c, n, b, l) AdaptiveParticleFilter(d, c, n, b, l), 'resultPrefix', 'adaptive'), ...
    struct('name', 'ResidualSquaredAdaptiveParticleFilter', 'factory', @(d, c, n, b, l) ResidualSquaredAdaptiveParticleFilter(d, c, n, b, l), 'resultPrefix', 'residualSq') ...
};
dataAll = loadSimulationData(h5File);
trainData = splitDataBySample(dataAll, 'train', opt.splitRatios, opt.splitSeed);
valData = splitDataBySample(dataAll, 'val', opt.splitRatios, opt.splitSeed);
testData = splitDataBySample(dataAll, 'test', opt.splitRatios, opt.splitSeed);

numNoise = numel(config.noiseVariance);

results = struct();
for filterIdx = 1:numel(targetFilters)
    filterSpec = targetFilters{filterIdx};

    bestBeta = nan(numNoise, 1);
    bestLambdaR = nan(numNoise, 1);
    bestValRMSE = inf(numNoise, 1);
    bestTrainRMSE = inf(numNoise, 1);

    fprintf('\n=== Per-noise Grid Search: %s ===\n', filterSpec.name);
    for noiseIdx = 1:numNoise
        fprintf('\n[Noise %.0e] %s beta-lambdaR search\n', config.noiseVariance(noiseIdx), filterSpec.name);

        for b = 1:numel(opt.betaGrid)
            for l = 1:numel(opt.lambdaRGrid)
                beta = opt.betaGrid(b);
                lambdaR = opt.lambdaRGrid(l);

                trainSeed = baseSeed + 3000 * noiseIdx + 100 * b + l;
                valSeed = baseSeed + 4000 * noiseIdx + 100 * b + l;

                factory = @(d, c, n) filterSpec.factory(d, c, n, beta, lambdaR);
                trainRMSE = evalSingleNoiseFilter(factory, trainData, config, noiseIdx, opt.tuneIterations, trainSeed);
                valRMSE = evalSingleNoiseFilter(factory, valData, config, noiseIdx, opt.tuneIterations, valSeed);

                fprintf('  beta=%.3f, lambdaR=%.3f -> train RMSE=%.6f, val RMSE=%.6f\n', ...
                    beta, lambdaR, trainRMSE, valRMSE);

                if valRMSE < bestValRMSE(noiseIdx)
                    bestValRMSE(noiseIdx) = valRMSE;
                    bestTrainRMSE(noiseIdx) = trainRMSE;
                    bestBeta(noiseIdx) = beta;
                    bestLambdaR(noiseIdx) = lambdaR;
                end
            end
        end
    end

    fprintf('\n=== Final Test Evaluation with Per-noise Best Params: %s ===\n', filterSpec.name);
    testRMSE = nan(numNoise, 1);
    testAPE = nan(numNoise, 1);

    for noiseIdx = 1:numNoise
        testFactory = @(d, c, n) filterSpec.factory(d, c, n, bestBeta(noiseIdx), bestLambdaR(noiseIdx));
        [testRMSE(noiseIdx), testAPE(noiseIdx)] = evalSingleNoiseFilterWithAPE( ...
            testFactory, testData, config, noiseIdx, opt.finalIterations, baseSeed + 5000 + noiseIdx);
    end

    fprintf('\n[Best Params] %s\n', filterSpec.name);
    fprintf('Noise | bestBeta | bestLambdaR | trainRMSE | valRMSE | testRMSE | testAPE\n');
    fprintf('--------------------------------------------------------------------------\n');
    for i = 1:numNoise
        fprintf('%.0e | %.4f | %.4f | %.6f | %.6f | %.6f | %.6f\n', ...
            config.noiseVariance(i), bestBeta(i), bestLambdaR(i), ...
            bestTrainRMSE(i), bestValRMSE(i), testRMSE(i), testAPE(i));
    end

    results.(filterSpec.resultPrefix).bestBeta = bestBeta;
    results.(filterSpec.resultPrefix).bestLambdaR = bestLambdaR;
    results.(filterSpec.resultPrefix).bestTrainRMSE = bestTrainRMSE;
    results.(filterSpec.resultPrefix).bestValRMSE = bestValRMSE;
    results.(filterSpec.resultPrefix).testRMSE = testRMSE;
    results.(filterSpec.resultPrefix).testAPE = testAPE;
end

resultFile = fullfile(config.pathResult, 'per_noise_gridsearch_AdaptiveParticleFilter_and_ResidualSquaredAdaptiveParticleFilter.mat');
if ~exist(config.pathResult, 'dir')
    mkdir(config.pathResult);
end
results.noiseVariance = config.noiseVariance;
results.options = opt;
save(resultFile, 'results');
fprintf('\nSaved results: %s\n', resultFile);

function dataOut = splitDataBySample(dataIn, splitName, splitRatios, splitSeed)
    splitName = lower(string(splitName));
    if numel(splitRatios) ~= 3
        error('splitRatios must be [train val test].');
    end

    splitRatios = splitRatios / sum(splitRatios);

    nSamples = size(dataIn.x_hat_LLS, 3);
    allIdx = 1:nSamples;

    if splitName == "all"
        idx = allIdx;
    else
        prev = rng;
        cleanupObj = onCleanup(@() rng(prev));
        rng(splitSeed, 'twister');
        perm = randperm(nSamples);
        clear cleanupObj;

        nTrain = floor(splitRatios(1) * nSamples);
        nVal = floor(splitRatios(2) * nSamples);
        nTest = nSamples - nTrain - nVal;

        trainIdx = sort(perm(1:nTrain));
        valIdx = sort(perm(nTrain + 1:nTrain + nVal));
        testIdx = sort(perm(nTrain + nVal + 1:nTrain + nVal + nTest));

        switch splitName
            case "train"
                idx = trainIdx;
            case "val"
                idx = valIdx;
            case "test"
                idx = testIdx;
            otherwise
                error('Unknown splitName: %s', char(splitName));
        end
    end

    dataOut = dataIn;
    dataOut.ranging = dataIn.ranging(:, :, idx, :);
    dataOut.x_hat_LLS = dataIn.x_hat_LLS(:, :, idx, :);
    dataOut.z_LLS = dataIn.z_LLS(:, :, idx, :);
    dataOut.R_LLS = dataIn.R_LLS(:, :, :, idx, :);
    dataOut.processNoise = dataIn.processNoise(:, idx, :);
    dataOut.toaNoise = dataIn.toaNoise(:, idx, :);

    dataOut.split.name = char(splitName);
    dataOut.split.indices = idx;
    dataOut.split.numSamples = numel(idx);
end

function rmse = evalSingleNoiseFilter(filterFactory, data, config, noiseIdx, numIterations, seed)
    [rmse, ~] = evalSingleNoiseFilterWithAPE(filterFactory, data, config, noiseIdx, numIterations, seed);
end

function [rmse, ape] = evalSingleNoiseFilterWithAPE(filterFactory, data, config, noiseIdx, numIterations, seed)
    rng(seed, 'twister');

    filterObj = filterFactory(data, config, noiseIdx);

    numPoints = size(data.x_hat_LLS, 2);
    maxIterAvail = size(data.x_hat_LLS, 3);
    nIter = min(numIterations, maxIterAvail);

    estNoise = zeros(2, numPoints, nIter);

    for iterIdx = 1:nIter
        state = filterObj.initializeState(numPoints);
        [state, p1, p2] = filterObj.initializeFirstTwo(state, iterIdx);
        estNoise(:, 1, iterIdx) = p1;
        estNoise(:, 2, iterIdx) = p2;

        for pointIdx = 3:numPoints
            [state, est] = filterObj.step(state, iterIdx, pointIdx);
            estNoise(:, pointIdx, iterIdx) = est;
        end
    end

    [rmse, ape] = evaluateFilter(estNoise, 3);
end
