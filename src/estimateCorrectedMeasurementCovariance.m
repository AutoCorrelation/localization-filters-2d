function stats = estimateCorrectedMeasurementCovariance(h5path, outMatPath, trajectoryName)
% ESTIMATECORRECTEDMEASUREMENTCOVARIANCE
% Estimate measurement-error covariance from /allRanging_corrected.
%
% Outputs include both 4x4 ranging covariance and 6x6 LLS-space covariance
% per noise level (global) and per point.
% Results are stored per-trajectory in one MAT file under `statsByTrajectory`.
%
% Usage:
%   stats = estimateCorrectedMeasurementCovariance();
%   stats = estimateCorrectedMeasurementCovariance('ranging_data_cv.h5');
%   stats = estimateCorrectedMeasurementCovariance('ranging_data_cv.h5', ...
%       fullfile('result','R_corrected_stats.mat'));
%   stats = estimateCorrectedMeasurementCovariance('ranging_data_cv.h5', ...
%       fullfile('result','R_corrected_stats.mat'), 'cv');

if nargin < 1 || isempty(h5path)
    srcDir = fileparts(mfilename('fullpath'));
    projectRoot = fileparts(srcDir);
    cand1 = fullfile(projectRoot, 'ranging_data_cv.h5');
    cand2 = fullfile(projectRoot, 'data', 'ranging_data_cv.h5');
    if exist(cand1, 'file')
        h5path = cand1;
    elseif exist(cand2, 'file')
        h5path = cand2;
    else
        error('estimateCorrectedMeasurementCovariance:MissingFile', ...
            'Could not find ranging_data_cv.h5. Please provide h5path.');
    end
end

if nargin < 2 || isempty(outMatPath)
    srcDir = fileparts(mfilename('fullpath'));
    projectRoot = fileparts(srcDir);
    outMatPath = fullfile(projectRoot, 'result', 'R_corrected_stats.mat');
end

if nargin < 3 || isempty(trajectoryName)
    trajectoryName = localInferTrajectoryName(h5path);
end
trajectoryKey = matlab.lang.makeValidName(lower(trajectoryName));

cfg = initializeConfig();
anchors = cfg.Anchor.'; % 4 x 2

info = h5info(h5path);
names = {info.Datasets.Name};
if ~any(strcmp(names, 'allRanging_corrected'))
    error('estimateCorrectedMeasurementCovariance:MissingDataset', ...
        'Dataset /allRanging_corrected not found in %s', h5path);
end
if ~any(strcmp(names, 'true_position'))
    error('estimateCorrectedMeasurementCovariance:MissingDataset', ...
        'Dataset /true_position not found in %s', h5path);
end

measCorr = h5read(h5path, '/allRanging_corrected');
truePos = h5read(h5path, '/true_position'); % 2 x P

rangingCorr = localToRanging(measCorr, anchors);

numAnchors = size(rangingCorr, 1);
numPoints = size(rangingCorr, 2);
numIter = size(rangingCorr, 3);
numNoise = size(rangingCorr, 4);

trueDist = localTrueDistances(truePos, anchors); % 4 x P
trueDist4D = repmat(trueDist, [1, 1, numIter, numNoise]);
errR = rangingCorr - trueDist4D; % 4 x P x T x N

% Global 4x4 covariance per noise (using all points and iterations).
R_corr_global = zeros(numAnchors, numAnchors, numNoise);
for n = 1:numNoise
    E = reshape(errR(:, :, :, n), numAnchors, []); % 4 x (P*T)
    R_corr_global(:, :, n) = localDiagVarCov(E);
end

% Per-point 4x4 covariance per noise (using only iteration axis).
R_corr_point = zeros(numAnchors, numAnchors, numPoints, numNoise);
for n = 1:numNoise
    for p = 1:numPoints
        Ep = squeeze(errR(:, p, :, n)); % 4 x T
        if size(Ep, 2) > 1
            R_corr_point(:, :, p, n) = localDiagVarCov(Ep);
        else
            R_corr_point(:, :, p, n) = zeros(numAnchors);
        end
    end
end

% Also estimate covariance in 6D LLS measurement space (z_LLS).
zCorr = localRangesToZ(rangingCorr);
zTrue = localRangesToZ(trueDist4D);
errZ = zCorr - zTrue; % 6 x P x T x N

R_lls_global = zeros(6, 6, numNoise);
for n = 1:numNoise
    Ez = reshape(errZ(:, :, :, n), 6, []);
    R_lls_global(:, :, n) = cov(Ez.');
end

R_lls_point = zeros(6, 6, numPoints, numNoise);
for n = 1:numNoise
    for p = 1:numPoints
        Ezp = squeeze(errZ(:, p, :, n)); % 6 x T
        if size(Ezp, 2) > 1
            R_lls_point(:, :, p, n) = cov(Ezp.');
        else
            R_lls_point(:, :, p, n) = zeros(6);
        end
    end
end

% Scalar summary for quick tuning (mean anchor variance per noise).
noiseVarEstimated = squeeze(mean(diag3(R_corr_global), 1)).';

stats = struct();
stats.h5path = h5path;
stats.numPoints = numPoints;
stats.numIterations = numIter;
stats.numNoise = numNoise;
stats.R_corrected_global = R_corr_global;   % 4x4xN
stats.R_corrected_point = R_corr_point;     % 4x4xPxN
stats.R_lls_global = R_lls_global;          % 6x6xN
stats.R_lls_point = R_lls_point;            % 6x6xPxN
stats.estimatedNoiseVariance = noiseVarEstimated;
stats.trajectory = trajectoryName;

outDir = fileparts(outMatPath);
if ~isempty(outDir) && ~exist(outDir, 'dir')
    mkdir(outDir);
end
% Store/merge by trajectory into one MAT file.
if exist(outMatPath, 'file')
    prev = load(outMatPath);
else
    prev = struct();
end

if isfield(prev, 'statsByTrajectory') && isstruct(prev.statsByTrajectory)
    statsByTrajectory = prev.statsByTrajectory;
else
    statsByTrajectory = struct();
end
statsByTrajectory.(trajectoryKey) = stats;

save(outMatPath, 'stats', 'statsByTrajectory');

fprintf('\n[estimateCorrectedMeasurementCovariance] Saved: %s\n', outMatPath);
fprintf('[estimateCorrectedMeasurementCovariance] trajectory: %s\n', trajectoryName);
fprintf('[estimateCorrectedMeasurementCovariance] estimatedNoiseVariance = %s\n', ...
    mat2str(noiseVarEstimated, 6));
end

function out = localToRanging(meas, anchors)
% Convert corrected measurement tensor to ranging tensor (4 x P x T x N).
if size(meas, 1) == 4
    out = meas;
    return;
end

if size(meas, 1) ~= 2
    error('localToRanging:UnsupportedShape', ...
        'Expected first dim 4 (range) or 2 (position), got %d', size(meas, 1));
end

numAnchors = size(anchors, 1);
numPoints = size(meas, 2);
numIter = size(meas, 3);
numNoise = size(meas, 4);

out = zeros(numAnchors, numPoints, numIter, numNoise);
for n = 1:numNoise
    for t = 1:numIter
        x = squeeze(meas(:, :, t, n)); % 2 x P
        for a = 1:numAnchors
            dx = x(1, :) - anchors(a, 1);
            dy = x(2, :) - anchors(a, 2);
            out(a, :, t, n) = sqrt(dx.^2 + dy.^2);
        end
    end
end
end

function d = localTrueDistances(truePos, anchors)
numAnchors = size(anchors, 1);
numPoints = size(truePos, 2);
d = zeros(numAnchors, numPoints);
for a = 1:numAnchors
    dx = truePos(1, :) - anchors(a, 1);
    dy = truePos(2, :) - anchors(a, 2);
    d(a, :) = sqrt(dx.^2 + dy.^2);
end
end

function z = localRangesToZ(r)
% r: 4 x P x T x N
numPoints = size(r, 2);
numIter = size(r, 3);
numNoise = size(r, 4);

r1 = squeeze(r(1, :, :, :));
r2 = squeeze(r(2, :, :, :));
r3 = squeeze(r(3, :, :, :));
r4 = squeeze(r(4, :, :, :));

z = zeros(6, numPoints, numIter, numNoise);
z(1, :, :, :) = r1.^2 - r2.^2 - 10^2;
z(2, :, :, :) = r1.^2 - r3.^2;
z(3, :, :, :) = r1.^2 - r4.^2 + 10^2;
z(4, :, :, :) = r2.^2 - r3.^2 + 10^2;
z(5, :, :, :) = r2.^2 - r4.^2 + 2 * 10^2;
z(6, :, :, :) = r3.^2 - r4.^2 + 10^2;
end

function d = diag3(M)
% Extract diagonals from 3D stack: A x A x N -> A x N
a = size(M, 1);
n = size(M, 3);
d = zeros(a, n);
for i = 1:n
    d(:, i) = diag(M(:, :, i));
end
end

function R = localDiagVarCov(E)
% E: d x n samples, returns d x d diagonal covariance from per-dimension variance
v = var(E, 0, 2);
v(~isfinite(v)) = 0;
R = diag(v);
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
