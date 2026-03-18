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
opt.betaGrid = [0.8, 0.9, 0.95, 0.98, 0.99, 0.995, 0.999];
opt.lambdaRGrid = [0.1, 0.5, 1, 2, 5, 10, 20, 50, 100];

h5File = fullfile(config.pathData, 'simulation_data.h5');
if ~isfile(h5File)
    dataGenerate(config);
end

dataAll = loadSimulationData(h5File);
trainData = splitDataBySample(dataAll, 'train', opt.splitRatios, opt.splitSeed);
valData = splitDataBySample(dataAll, 'val', opt.splitRatios, opt.splitSeed);
testData = splitDataBySample(dataAll, 'test', opt.splitRatios, opt.splitSeed);

numNoise = numel(config.noiseVariance);

bestBeta = nan(numNoise, 1);
bestLambdaR = nan(numNoise, 1);
bestAdaptiveValRMSE = inf(numNoise, 1);
bestAdaptiveTrainRMSE = inf(numNoise, 1);

fprintf('\n=== Per-noise Grid Search: AdaptiveParticleFilter ===\n');
for noiseIdx = 1:numNoise
    fprintf('\n[Noise %.0e] beta-lambdaR search\n', config.noiseVariance(noiseIdx));

    for b = 1:numel(opt.betaGrid)
        for l = 1:numel(opt.lambdaRGrid)
            beta = opt.betaGrid(b);
            lambdaR = opt.lambdaRGrid(l);

            trainSeed = baseSeed + 3000 * noiseIdx + 100 * b + l;
            valSeed = baseSeed + 4000 * noiseIdx + 100 * b + l;

            factory = @(d, c, n) AdaptiveParticleFilter(d, c, n, beta, lambdaR);
            trainRMSE = evalSingleNoiseFilter(factory, trainData, config, noiseIdx, opt.tuneIterations, trainSeed);
            valRMSE = evalSingleNoiseFilter(factory, valData, config, noiseIdx, opt.tuneIterations, valSeed);

            fprintf('  beta=%.3f, lambdaR=%.3f -> train RMSE=%.6f, val RMSE=%.6f\n', ...
                beta, lambdaR, trainRMSE, valRMSE);

            if valRMSE < bestAdaptiveValRMSE(noiseIdx)
                bestAdaptiveValRMSE(noiseIdx) = valRMSE;
                bestAdaptiveTrainRMSE(noiseIdx) = trainRMSE;
                bestBeta(noiseIdx) = beta;
                bestLambdaR(noiseIdx) = lambdaR;
            end
        end
    end
end

% Final test evaluation with selected per-noise parameters
fprintf('\n=== Final Test Evaluation with Per-noise Best Params ===\n');

adaptiveTestRMSE = nan(numNoise, 1);
adaptiveTestMAE = nan(numNoise, 1);

for noiseIdx = 1:numNoise
    adaptiveFactory = @(d, c, n) AdaptiveParticleFilter(d, c, n, bestBeta(noiseIdx), bestLambdaR(noiseIdx));
    [adaptiveTestRMSE(noiseIdx), adaptiveTestMAE(noiseIdx)] = evalSingleNoiseFilterWithMAE( ...
        adaptiveFactory, testData, config, noiseIdx, opt.finalIterations, baseSeed + 5000 + noiseIdx);
end

fprintf('\n[Best Params] AdaptiveParticleFilter\n');
fprintf('Noise | bestBeta | bestLambdaR | trainRMSE | valRMSE | testRMSE | testMAE\n');
fprintf('--------------------------------------------------------------------------\n');
for i = 1:numNoise
    fprintf('%.0e | %.4f | %.4f | %.6f | %.6f | %.6f | %.6f\n', ...
        config.noiseVariance(i), bestBeta(i), bestLambdaR(i), ...
        bestAdaptiveTrainRMSE(i), bestAdaptiveValRMSE(i), adaptiveTestRMSE(i), adaptiveTestMAE(i));
end

resultFile = fullfile(config.pathResult, 'per_noise_gridsearch_results.mat');
if ~exist(config.pathResult, 'dir')
    mkdir(config.pathResult);
end
results = struct();
results.bestBeta = bestBeta;
results.bestLambdaR = bestLambdaR;
results.bestAdaptiveTrainRMSE = bestAdaptiveTrainRMSE;
results.bestAdaptiveValRMSE = bestAdaptiveValRMSE;
results.adaptiveTestRMSE = adaptiveTestRMSE;
results.adaptiveTestMAE = adaptiveTestMAE;
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
    [rmse, ~] = evalSingleNoiseFilterWithMAE(filterFactory, data, config, noiseIdx, numIterations, seed);
end

function [rmse, mae] = evalSingleNoiseFilterWithMAE(filterFactory, data, config, noiseIdx, numIterations, seed)
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

    [rmse, mae] = evaluateFilter(estNoise, 3);
end
