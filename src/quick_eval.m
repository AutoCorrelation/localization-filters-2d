function results = quick_eval(h5path, opts)
% QUICK_EVAL  Run a quick PF benchmark from measurement-only H5 data.
%
% This script is designed for files like `ranging_data_cv.h5` that contain
% `/allRanging`, `/distance`, and `/true_position`.
% It mimics the `main_sub.m` execution flow:
%   1) Build filter input tensors from measurement data
%   2) Run selected filters via `runFilter`
%   3) Report per-noise RMSE and plot comparison
%
% Usage examples:
%   quick_eval();
%   quick_eval('ranging_data_cv.h5');
%   quick_eval('ranging_data_cv.h5', struct('numParticles', 500, ...
%       'filterList', {{'Baseline'; 'NonlinearParticleFilter'}}));
%
% ------------------------- Tunable Parameters -------------------------
% Pass these fields in `opts` to tune quickly:
%   opts.numParticles        : particle count for PF (default: 1000)
%   opts.maxIterations       : max iterations to use from H5 (default: 1000)
%   opts.filterList          : filters to run (default: Baseline + NonlinearPF)
%   opts.noiseVariance       : noise variance vector (default: initializeConfig)
%   opts.plotResults         : plot RMSE/runtime figure (default: true)
%   opts.showReferenceRMSE   : compute RMSE from /estimatedPF if available
%                              (default: true)
%   opts.compareCorrected    : compare /allRanging_corrected too (default: true)
%   opts.RCorrectedPoint     : 4x4xP xN covariance for corrected scenario
%   opts.trajectoryName      : 'cv' | 'circular' | 'zigzag' (optional)
%   opts.disablePredictNoiseCorrected : disable process-noise sampling in
%                                       corrected scenario (default: false)
% --------------------------------------------------------------------

if nargin < 1
    h5path = '';
end
if nargin < 2
    opts = struct();
end

srcDir = fileparts(mfilename('fullpath'));
addpath(fullfile(srcDir, 'utils'));

defaults = localDefaultOptions();
opts = localMergeStruct(defaults, opts);

if isempty(h5path)
    projectRoot = fileparts(srcDir);
    cand1 = fullfile(projectRoot, 'ranging_data_cv.h5');
    cand2 = fullfile(projectRoot, 'data', 'ranging_data_cv.h5');
    if exist(cand1, 'file')
        h5path = cand1;
    elseif exist(cand2, 'file')
        h5path = cand2;
    else
        error('quick_eval:MissingFile', ...
            'H5 file not found. Provide explicit path to quick_eval(h5path).');
    end
end

fprintf('\n[quick_eval] H5 file: %s\n', h5path);
fprintf('[quick_eval] numParticles=%d, maxIterations=%d\n', ...
    round(opts.numParticles), round(opts.maxIterations));
fprintf('[quick_eval] filters: %s\n', strjoin(opts.filterList(:).', ', '));

cfg = initializeConfig(opts.numParticles);

if ~isempty(opts.noiseVariance)
    cfg.noiseVariance = opts.noiseVariance(:).';
end

raw = localReadMeasurementH5(h5path);
hasCorrected = opts.compareCorrected && ~isempty(raw.allRanging_corrected);
doScenarioPlot = opts.plotResults && ~hasCorrected;

baseRun = localRunScenario(raw.allRanging, 'allRanging', raw, cfg, opts, doScenarioPlot);

if opts.showReferenceRMSE && ~isempty(raw.estimatedPF)
    refRmse = localComputeReferenceRmse(raw.estimatedPF, raw.true_position, baseRun.config.iterations);
    fprintf('[quick_eval] Reference /estimatedPF RMSE (mean over points): %.6g\n', refRmse);
end

results = struct();
results.h5path = h5path;
results.filterList = opts.filterList;
results.base = baseRun;

if hasCorrected
    rCorrPoint = localResolveCorrectedRPoint(opts, raw, cfg, h5path);
    corrRun = localRunScenario(raw.allRanging_corrected, 'allRanging_corrected', raw, cfg, opts, doScenarioPlot, rCorrPoint, opts.disablePredictNoiseCorrected);
    results.corrected = corrRun;

    numNoise = size(baseRun.rmseMat, 1);
    numFilters = numel(opts.filterList);
    varNames = {'NoiseVariance'};
    outMat = baseRun.config.noiseVariance(:);
    for k = 1:numFilters
        vBase = matlab.lang.makeValidName(opts.filterList{k});
        vCorr = matlab.lang.makeValidName([opts.filterList{k} '_corrected']);
        varNames = [varNames, {vBase, vCorr}]; %#ok<AGROW>
        outMat = [outMat, ...
            baseRun.rmseMat(1:numNoise, k), ...
            corrRun.rmseMat(1:numNoise, k)]; %#ok<AGROW>
    end

    results.comparisonTable = array2table(outMat, 'VariableNames', varNames);
    disp(results.comparisonTable);

    if opts.plotResults
        localPlotComparisonSemilogX(baseRun.config.noiseVariance, baseRun.rmseMat, corrRun.rmseMat, opts.filterList);
    end
else
    results.comparisonTable = baseRun.rmseTable;
    if opts.plotResults
        localPlotSingleSemilogX(baseRun.config.noiseVariance, baseRun.rmseMat, opts.filterList, baseRun.scenarioName);
    end
end

fprintf('[quick_eval] Completed.\n\n');
end

function opts = localDefaultOptions()
opts = struct();
opts.numParticles = 1e3;
opts.maxIterations = 1e3;
opts.filterList = {
    'Baseline';
    'NonlinearParticleFilter'};
opts.noiseVariance = [];
opts.plotResults = true;
opts.showReferenceRMSE = true;
opts.compareCorrected = true;
opts.RCorrectedPoint = [];
opts.trajectoryName = '';
opts.disablePredictNoiseCorrected = false;
end

function out = localReadMeasurementH5(h5path)
info = h5info(h5path);
names = {info.Datasets.Name};

if ~any(strcmp(names, 'allRanging'))
    error('quick_eval:MissingDataset', 'Missing required dataset `/allRanging`.');
end
if ~any(strcmp(names, 'true_position'))
    error('quick_eval:MissingDataset', 'Missing required dataset `/true_position`.');
end

out = struct();
out.info = info;
out.allRanging = h5read(h5path, '/allRanging');
out.true_position = h5read(h5path, '/true_position');
if any(strcmp(names, 'allRanging_corrected'))
    out.allRanging_corrected = h5read(h5path, '/allRanging_corrected');
else
    out.allRanging_corrected = [];
end

if any(strcmp(names, 'distance'))
    out.distance = h5read(h5path, '/distance');
else
    out.distance = [];
end

if any(strcmp(names, 'estimatedPF'))
    out.estimatedPF = h5read(h5path, '/estimatedPF');
else
    out.estimatedPF = [];
end
end

function runOut = localRunScenario(allR, scenarioName, raw, cfgIn, opts, doPlot, rCorrPoint, disablePredictNoise)
if nargin < 7
    rCorrPoint = [];
end
if nargin < 8
    disablePredictNoise = false;
end

[data, cfg] = localBuildFilterInputFromMeasurement(allR, raw.true_position, raw.distance, cfgIn, opts.maxIterations);
if ~isempty(rCorrPoint)
    data.R_corrected_point = rCorrPoint;
end
if disablePredictNoise
    cfg.disableProcessNoise = true;
end

numFilters = numel(opts.filterList);
numNoise = numel(cfg.noiseVariance);
metrics = cell(numFilters, 1);
filterTimes = zeros(numFilters, 1);

fprintf('[quick_eval] Scenario: %s\n', scenarioName);
for k = 1:numFilters
    fname = opts.filterList{k};
    fprintf('[quick_eval] Running %s (%s) ... ', fname, scenarioName);
    tStart = tic;
    try
        [~, out] = runFilter(fname, data, cfg);
        metrics{k} = out;
        filterTimes(k) = toc(tStart);
        fprintf('done (mean RMSE=%.6g, %.3fs)\n', mean(out.RMSE), filterTimes(k));
    catch ME
        metrics{k} = struct('RMSE', nan(numNoise, 1), 'APE', nan(numNoise, 1));
        filterTimes(k) = toc(tStart);
        fprintf('failed: %s\n', ME.message);
    end
end

rmseMat = nan(numNoise, numFilters);
for k = 1:numFilters
    rmseVec = metrics{k}.RMSE(:);
    rmseMat(:, k) = rmseVec(1:min(numNoise, numel(rmseVec)));
end

runOut = struct();
runOut.scenarioName = scenarioName;
runOut.config = cfg;
runOut.rmseMat = rmseMat;
runOut.runtimeSec = filterTimes;
runOut.rmseTable = array2table([cfg.noiseVariance(:), rmseMat], ...
    'VariableNames', [{'NoiseVariance'}, opts.filterList(:).']);
disp(runOut.rmseTable);

if doPlot
    localPlotSingleSemilogX(cfg.noiseVariance, rmseMat, opts.filterList, scenarioName);
end
end

function localPlotComparisonSemilogX(noiseVariance, rmseBase, rmseCorr, filterList)
figure('Name', 'RMSE Comparison: base vs corrected', 'NumberTitle', 'off');
hold on;

numFilters = numel(filterList);
lineHandles = gobjects(2 * numFilters, 1);
legendNames = cell(2 * numFilters, 1);

idx = 1;
for k = 1:numFilters
    lineHandles(idx) = semilogx(noiseVariance, rmseBase(:, k), '-o', 'LineWidth', 1.5);
    legendNames{idx} = filterList{k};
    idx = idx + 1;

    lineHandles(idx) = semilogx(noiseVariance, rmseCorr(:, k), '--s', 'LineWidth', 1.5);
    legendNames{idx} = [filterList{k} '(Corrected)'];
    idx = idx + 1;
end

xlabel('Noise Variance');
ylabel('RMSE');
title('RMSE by Noise Variance (Base vs Corrected)');
legend(lineHandles, legendNames, 'Location', 'northwest');
set(gca, 'XScale', 'log');
grid on;
hold off;
end

function localPlotSingleSemilogX(noiseVariance, rmseMat, filterList, scenarioName)
figure('Name', ['RMSE semilogx: ' scenarioName], 'NumberTitle', 'off');
semilogx(noiseVariance, rmseMat, 'LineWidth', 1.5);
set(gca, 'XScale', 'log');
legend(filterList, 'Location', 'northwest');
xlabel('Noise Variance');
ylabel('RMSE');
title(['RMSE by Noise Variance (' scenarioName ')']);
grid on;
end

function [data, cfg] = localBuildFilterInputFromMeasurement(measIn, truePos, distance, cfg, maxIterations)

numChannels = size(measIn, 1);
if numChannels == 4
    allR = measIn;
    posMeas = [];
elseif numChannels == 2
    % corrected position tensor -> synthesize ranging to anchors
    posMeas = measIn;
    allR = localPositionToRanging(posMeas, cfg.Anchor);
else
    error('quick_eval:UnsupportedMeasurementShape', ...
        'Expected first dimension 4 (ranging) or 2 (position), got %d.', numChannels);
end

numPoints = size(allR, 2);
numIterAvail = size(allR, 3);
numNoise = size(allR, 4);
numIterUse = min(maxIterations, numIterAvail);

allR = allR(:, :, 1:numIterUse, :);
if ~isempty(posMeas)
    posMeas = posMeas(:, :, 1:numIterUse, :);
end

if numel(cfg.noiseVariance) ~= numNoise
    cfg.noiseVariance = 1:numNoise;
end
cfg.iterations = numIterUse;

% Build LLS-form measurement tensors from 4-anchor ranging data.
z_LLS = zeros(6, numPoints, numIterUse, numNoise);
R_LLS = zeros(6, 6, numPoints, numIterUse, numNoise);
x_hat_LLS = zeros(2, numPoints, numIterUse, numNoise);

pinvH = cfg.pinvH;

for n = 1:numNoise
    nv = cfg.noiseVariance(n);

    r1 = squeeze(allR(1, :, :, n));
    r2 = squeeze(allR(2, :, :, n));
    r3 = squeeze(allR(3, :, :, n));
    r4 = squeeze(allR(4, :, :, n));

    z = zeros(6, numPoints, numIterUse);
    z(1, :, :) = r1.^2 - r2.^2 - 10^2;
    z(2, :, :) = r1.^2 - r3.^2;
    z(3, :, :) = r1.^2 - r4.^2 + 10^2;
    z(4, :, :) = r2.^2 - r3.^2 + 10^2;
    z(5, :, :) = r2.^2 - r4.^2 + 2 * 10^2;
    z(6, :, :) = r3.^2 - r4.^2 + 10^2;
    z_LLS(:, :, :, n) = z;

    if ~isempty(posMeas)
        x_hat_LLS(:, :, :, n) = posMeas(:, :, :, n);
    else
        z2 = reshape(z, 6, []);
        x2 = pinvH * z2;
        x_hat_LLS(:, :, :, n) = reshape(x2, 2, numPoints, numIterUse);
    end

    r1s = r1.^2; r2s = r2.^2; r3s = r3.^2; r4s = r4.^2;

    R_LLS(1, 1, :, :, n) = 4 * nv * (r1s + r2s);
    R_LLS(1, 2, :, :, n) = 4 * nv * r1s;
    R_LLS(1, 3, :, :, n) = 4 * nv * r1s;
    R_LLS(1, 4, :, :, n) = -4 * nv * r2s;
    R_LLS(1, 5, :, :, n) = -4 * nv * r2s;

    R_LLS(2, 1, :, :, n) = 4 * nv * r1s;
    R_LLS(2, 2, :, :, n) = 4 * nv * (r1s + r3s);
    R_LLS(2, 3, :, :, n) = 4 * nv * r1s;
    R_LLS(2, 4, :, :, n) = 4 * nv * r3s;
    R_LLS(2, 6, :, :, n) = -4 * nv * r3s;

    R_LLS(3, 1, :, :, n) = 4 * nv * r1s;
    R_LLS(3, 2, :, :, n) = 4 * nv * r1s;
    R_LLS(3, 3, :, :, n) = 4 * nv * (r1s + r4s);
    R_LLS(3, 5, :, :, n) = 4 * nv * r4s;
    R_LLS(3, 6, :, :, n) = 4 * nv * r4s;

    R_LLS(4, 1, :, :, n) = -4 * nv * r2s;
    R_LLS(4, 2, :, :, n) = 4 * nv * r3s;
    R_LLS(4, 4, :, :, n) = 4 * nv * (r2s + r3s);
    R_LLS(4, 5, :, :, n) = 4 * nv * r2s;
    R_LLS(4, 6, :, :, n) = -4 * nv * r3s;

    R_LLS(5, 1, :, :, n) = -4 * nv * r2s;
    R_LLS(5, 3, :, :, n) = 4 * nv * r4s;
    R_LLS(5, 4, :, :, n) = 4 * nv * r2s;
    R_LLS(5, 5, :, :, n) = 4 * nv * (r2s + r4s);
    R_LLS(5, 6, :, :, n) = 4 * nv * r4s;

    R_LLS(6, 2, :, :, n) = -4 * nv * r3s;
    R_LLS(6, 3, :, :, n) = 4 * nv * r4s;
    R_LLS(6, 4, :, :, n) = -4 * nv * r3s;
    R_LLS(6, 5, :, :, n) = 4 * nv * r4s;
    R_LLS(6, 6, :, :, n) = 4 * nv * (r3s + r4s);
end

% Build PF noise-bank fields using the same pattern as data generation code.
point2 = min(2, numPoints);
point3 = min(3, numPoints);

vel = squeeze(x_hat_LLS(:, point2, :, :) - x_hat_LLS(:, 1, :, :));
processNoise = truePos(:, point3) - squeeze(x_hat_LLS(:, point2, :, :)) - vel;
toaNoise = truePos(:, point2) - squeeze(x_hat_LLS(:, point2, :, :));

if ndims(processNoise) == 2
    processNoise = reshape(processNoise, size(processNoise, 1), size(processNoise, 2), 1);
    toaNoise = reshape(toaNoise, size(toaNoise, 1), size(toaNoise, 2), 1);
end

processbias = squeeze(mean(processNoise, 2));
if isvector(processbias)
    processbias = reshape(processbias, [2, 1]);
end

Q = zeros(2, 2, numNoise);
P0 = zeros(2, 2, numNoise);
for n = 1:numNoise
    pn = processNoise(:, :, n);
    tn = toaNoise(:, :, n);

    pb = mean(pn, 2);
    tb = mean(tn, 2);

    Q(:, :, n) = (pn * pn.') / size(pn, 2) - pb * pb.';
    P0(:, :, n) = (tn * tn.') / size(tn, 2) - tb * tb.';
end

% true_state shape expected by runFilter: (>=2, numPoints, numIterations, numNoise)
true_state = repmat(truePos, [1, 1, numIterUse, numNoise]);

data = struct();
data.allRanging = allR;
data.ranging = allR;
data.distance = distance;
data.true_position = truePos;

data.x_hat_LLS = x_hat_LLS;
data.z_LLS = z_LLS;
data.R_LLS = R_LLS;
data.Q = Q;
data.P0 = P0;
data.processNoise = processNoise;
data.toaNoise = toaNoise;
data.processbias = processbias;
data.mode_history = ones(numPoints, numIterUse, numNoise);
data.true_state = true_state;
end

function allR = localPositionToRanging(posTensor, anchor2xM)
% posTensor: 2 x P x T x N
% anchor2xM: 2 x M
anchors = anchor2xM.'; % M x 2
numAnchors = size(anchors, 1);
numPoints = size(posTensor, 2);
numIter = size(posTensor, 3);
numNoise = size(posTensor, 4);

allR = zeros(numAnchors, numPoints, numIter, numNoise);
for n = 1:numNoise
    for t = 1:numIter
        x = squeeze(posTensor(:, :, t, n)); % 2 x P
        for a = 1:numAnchors
            dx = x(1, :) - anchors(a, 1);
            dy = x(2, :) - anchors(a, 2);
            allR(a, :, t, n) = sqrt(dx.^2 + dy.^2);
        end
    end
end
end

function rmse = localComputeReferenceRmse(estimatedPF, truePos, numIterUse)
est = estimatedPF(:, :, 1:min(numIterUse, size(estimatedPF, 3)));
estMean = mean(est, 3);
d = estMean - truePos;
rmse = sqrt(mean(sum(d.^2, 1)));
end

function merged = localMergeStruct(base, override)
merged = base;
if isempty(override)
    return;
end

fields = fieldnames(override);
for i = 1:numel(fields)
    merged.(fields{i}) = override.(fields{i});
end
end

function Rcorr = localResolveCorrectedRPoint(opts, raw, cfg, h5path)
if isfield(opts, 'RCorrectedPoint') && ~isempty(opts.RCorrectedPoint)
    Rcorr = opts.RCorrectedPoint;
    return;
end

srcDir = fileparts(mfilename('fullpath'));
projectRoot = fileparts(srcDir);
matPath = fullfile(projectRoot, 'result', 'R_corrected_stats.mat');
if exist(matPath, 'file')
    loaded = load(matPath);
    if isfield(opts, 'trajectoryName') && ~isempty(opts.trajectoryName)
        traj = lower(opts.trajectoryName);
    else
        traj = localInferTrajectoryName(h5path);
    end
    trajKey = matlab.lang.makeValidName(traj);

    if isfield(loaded, 'statsByTrajectory') && isstruct(loaded.statsByTrajectory)
        if isfield(loaded.statsByTrajectory, trajKey)
            S = loaded.statsByTrajectory.(trajKey);
            if isfield(S, 'R_corrected_point')
                Rcorr = S.R_corrected_point;
                return;
            end
        end
    end

    if isfield(loaded, 'stats') && isfield(loaded.stats, 'R_corrected_point')
        Rcorr = loaded.stats.R_corrected_point;
        return;
    end
end

% Fallback: estimate quickly from corrected ranging and true positions.
rCorr = raw.allRanging_corrected;
if size(rCorr, 1) ~= 4
    rCorr = localPositionToRanging(rCorr, cfg.Anchor);
end

function name = localInferTrajectoryName(h5path)
base = lower(h5path);
if contains(base, 'circular')
    name = 'circular';
elseif contains(base, 'zigzag')
    name = 'zigzag';
elseif contains(base, 'cv')
    name = 'cv';
else
    name = 'cv';
end
end

numAnchors = size(rCorr, 1);
numPoints = size(rCorr, 2);
numNoise = size(rCorr, 4);

anchors = cfg.Anchor.';
trueDist = zeros(numAnchors, numPoints);
for a = 1:numAnchors
    dx = raw.true_position(1, :) - anchors(a, 1);
    dy = raw.true_position(2, :) - anchors(a, 2);
    trueDist(a, :) = sqrt(dx.^2 + dy.^2);
end

Rcorr = zeros(numAnchors, numAnchors, numPoints, numNoise);
for n = 1:numNoise
    for p = 1:numPoints
        E = squeeze(rCorr(:, p, :, n) - trueDist(:, p)); % 4 x T
        if size(E, 2) > 1
            v = var(E, 0, 2);
            v(~isfinite(v)) = 0;
            Rcorr(:, :, p, n) = diag(v);
        else
            Rcorr(:, :, p, n) = zeros(numAnchors);
        end
    end
end
end
