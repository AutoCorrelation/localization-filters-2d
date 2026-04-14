function dataGenerateIMM(config)
% DATAGENERATEIMM Generate IMM(CV/CT) trajectory-based data in the same H5 schema.
% Output dataset names/shapes are kept compatible with loadSimulationData/runFilter.

pathData = config.pathData;
pathResult = config.pathResult;
numSamples = config.numSamples;
noiseVariance = config.noiseVariance;
numPoints = config.numPoints;
Anchor = config.Anchor;
pinvH = config.pinvH;

if ~exist(pathData, 'dir')
    mkdir(pathData)
end
if ~exist(pathResult, 'dir')
    mkdir(pathResult)
end

numNoises = numel(noiseVariance);

% IMM motion-model parameters
if ~isfield(config, 'immDt')
    config.immDt = 1.0;
end
if ~isfield(config, 'immSigmaV')
    config.immSigmaV = 0.5;
end
if ~isfield(config, 'immOmegas')
    config.immOmegas = [0, 0.15, -0.15];
end
if ~isfield(config, 'immTPM')
    config.immTPM = [0.90, 0.05, 0.05; 0.10, 0.90, 0.00; 0.10, 0.00, 0.90];
end
if ~isfield(config, 'immInitialModeProb')
    config.immInitialModeProb = [1; 0; 0];
end
if ~isfield(config, 'immInitialState')
    config.immInitialState = [0; 0; 10; 10];
end
if ~isfield(config, 'immMotionScale')
    % Conservative default scale to keep trajectory naturally inside anchor bounds.
    config.immMotionScale = 0.07;
end
if ~isfield(config, 'immProposalMaxTry')
    % Per-step sampling retries before declaring the trajectory invalid.
    config.immProposalMaxTry = 60;
end
if ~isfield(config, 'immTrajectoryMaxRetry')
    % Number of times to regenerate a full trajectory if any step exits bounds.
    config.immTrajectoryMaxRetry = 200;
end

omegas = config.immOmegas(:).';
TPM = config.immTPM;
initialModeProb = config.immInitialModeProb(:);
x0 = config.immInitialState(:);
% Force start point to (0,0) as requested.
x0(1:2) = [0; 0];

motionScale = max(config.immMotionScale, 1e-6);
x0(3:4) = x0(3:4) * motionScale;

% Keep IMM trajectory inside anchor bounding box.
anchorMin = min(Anchor, [], 2);
anchorMax = max(Anchor, [], 2);

dt = config.immDt;
sigmaV = config.immSigmaV * motionScale;
proposalMaxTry = max(1, round(config.immProposalMaxTry));
trajectoryMaxRetry = max(1, round(config.immTrajectoryMaxRetry));
Q4 = [dt^3/3, 0, dt^2/2, 0; ...
      0, dt^3/3, 0, dt^2/2; ...
      dt^2/2, 0, dt, 0; ...
      0, dt^2/2, 0, dt] * sigmaV^2;

% Preallocate output tensors (same format as dataGenerate.m)
ranging_cell = cell(numNoises, 1);
x_hat_LLS_cell = cell(numNoises, 1);
z_LLS_cell = cell(numNoises, 1);
R_LLS_cell = cell(numNoises, 1);

% Optional debug/analysis tensors
true_state_cell = cell(numNoises, 1);
mode_history_cell = cell(numNoises, 1);

% Generate one global random true trajectory shared across all noise levels.
isValidGlobalTrajectory = false;
for retryIdx = 1:trajectoryMaxRetry
    [globalTrueStates, globalModeHistory, isValidGlobalTrajectory] = localGenerateOneIMMRun( ...
        numPoints, x0, dt, omegas, TPM, initialModeProb, Q4, anchorMin, anchorMax, proposalMaxTry);
    if isValidGlobalTrajectory
        break;
    end
end
if ~isValidGlobalTrajectory
    error('dataGenerateIMM:GlobalTrajectoryGenerationFailed', ...
        'Failed to generate a shared in-bounds IMM trajectory after %d retries.', trajectoryMaxRetry);
end

parfor n = 1:numNoises
    noiseVar = noiseVariance(n);

    ranging_temp = zeros(4, numPoints, numSamples);
    z_LLS_temp = zeros(6, numPoints, numSamples);
    R_LLS_temp = zeros(6, 6, numPoints, numSamples);
    x_hat_LLS_temp = zeros(2, numPoints, numSamples);

    true_state_temp = zeros(4, numPoints, numSamples);
    mode_history_temp = zeros(numPoints, numSamples);

    % For each sample: reuse one shared true trajectory and add only measurement noise.
    for s = 1:numSamples
        trueStates = globalTrueStates;
        modeHistory = globalModeHistory;

        % Build ranging/LLS observations from generated trajectory
        for k = 1:numPoints
            pTrue = trueStates(1:2, k);
            d = vecnorm(pTrue - Anchor, 2, 1).';
            ranging_temp(:, k, s) = d + sqrt(noiseVar) * randn(4, 1);

            r1 = ranging_temp(1, k, s);
            r2 = ranging_temp(2, k, s);
            r3 = ranging_temp(3, k, s);
            r4 = ranging_temp(4, k, s);

            z_LLS_temp(:, k, s) = [ ...
                r1^2 - r2^2 - 10^2; ...
                r1^2 - r3^2; ...
                r1^2 - r4^2 + 10^2; ...
                r2^2 - r3^2 + 10^2; ...
                r2^2 - r4^2 + 2 * 10^2; ...
                r3^2 - r4^2 + 10^2];

            R_LLS_temp(:, :, k, s) = [ ...
                4*noiseVar*(r1^2+r2^2), 4*noiseVar*(r1^2), 4*noiseVar*(r1^2), -4*noiseVar*(r2^2), -4*noiseVar*(r2^2), 0; ...
                4*noiseVar*(r1^2), 4*noiseVar*(r1^2+r3^2), 4*noiseVar*(r1^2), 4*noiseVar*(r3^2), 0, -4*noiseVar*(r3^2); ...
                4*noiseVar*(r1^2), 4*noiseVar*(r1^2), 4*noiseVar*(r1^2+r4^2), 0, 4*noiseVar*(r4^2), 4*noiseVar*(r4^2); ...
                -4*noiseVar*(r2^2), 4*noiseVar*(r3^2), 0, 4*noiseVar*(r2^2+r3^2), 4*noiseVar*(r2^2), -4*noiseVar*(r3^2); ...
                -4*noiseVar*(r2^2), 0, 4*noiseVar*(r4^2), 4*noiseVar*(r2^2), 4*noiseVar*(r2^2+r4^2), 4*noiseVar*(r4^2); ...
                0, -4*noiseVar*(r3^2), 4*noiseVar*(r4^2), -4*noiseVar*(r3^2), 4*noiseVar*(r4^2), 4*noiseVar*(r3^2+r4^2)];

            x_hat_LLS_temp(:, k, s) = pinvH * z_LLS_temp(:, k, s);
        end

        true_state_temp(:, :, s) = trueStates;
        mode_history_temp(:, s) = modeHistory(:);
    end

    ranging_cell{n} = ranging_temp;
    z_LLS_cell{n} = z_LLS_temp;
    R_LLS_cell{n} = R_LLS_temp;
    x_hat_LLS_cell{n} = x_hat_LLS_temp;

    true_state_cell{n} = true_state_temp;
    mode_history_cell{n} = mode_history_temp;
end

ranging = cat(4, ranging_cell{:});
x_hat_LLS = cat(4, x_hat_LLS_cell{:});
z_LLS = cat(4, z_LLS_cell{:});
R_LLS = cat(5, R_LLS_cell{:});

true_state = cat(4, true_state_cell{:});
mode_history = cat(3, mode_history_cell{:});

% Match CV pipeline for Q/P0 estimation from x_hat trajectory statistics.
vel = x_hat_LLS(:, 2, :, :) - x_hat_LLS(:, 1, :, :);
vel = squeeze(vel); % 2 x numSamples x numNoises

processNoise = zeros(2, numSamples, numNoises);
toaNoise = zeros(2, numSamples, numNoises);
for n = 1:numNoises
    x2 = squeeze(x_hat_LLS(:, 2, :, n));        % 2 x numSamples
    velN = squeeze(vel(:, :, n));               % 2 x numSamples
    p3 = squeeze(true_state(1:2, 3, 1, n));     % 2 x 1
    p2 = squeeze(true_state(1:2, 2, 1, n));     % 2 x 1

    processNoise(:, :, n) = p3 - x2 - velN;
    toaNoise(:, :, n) = p2 - x2;
end

eeT_all = cell(numNoises, 1);
xxT_all = cell(numNoises, 1);
parfor n = 1:numNoises
    eeT_n = zeros(2, 2, numSamples);
    xxT_n = zeros(2, 2, numSamples);

    for i = 1:numSamples
        eeT_n(:, :, i) = processNoise(:, i, n) * processNoise(:, i, n)';
        xxT_n(:, :, i) = toaNoise(:, i, n) * toaNoise(:, i, n)';
    end

    eeT_all{n} = eeT_n;
    xxT_all{n} = xxT_n;
end

eeT = cat(4, eeT_all{:});
xxT = cat(4, xxT_all{:});
EeeT = squeeze(mean(eeT, 3));
ExxT = squeeze(mean(xxT, 3));

processbias = squeeze(mean(processNoise, 2));
toabias = squeeze(mean(toaNoise, 2));

Q = zeros(2, 2, numNoises);
P0 = zeros(2, 2, numNoises);
for n = 1:numNoises
    Q(:, :, n) = EeeT(:, :, n) - processbias(:, n) * processbias(:, n)';
    P0(:, :, n) = ExxT(:, :, n) - toabias(:, n) * toabias(:, n)';
end

% Choose h5 filename based on motion model
h5FileName = 'simulation_data_imm.h5';
h5File = fullfile(pathData, h5FileName);
if isfile(h5File)
    delete(h5File)
end

h5create(h5File, '/ranging', size(ranging), 'DataType', 'double');
h5write(h5File, '/ranging', ranging);

h5create(h5File, '/x_hat_LLS', size(x_hat_LLS), 'DataType', 'double');
h5write(h5File, '/x_hat_LLS', x_hat_LLS);

h5create(h5File, '/z_LLS', size(z_LLS), 'DataType', 'double');
h5write(h5File, '/z_LLS', z_LLS);

h5create(h5File, '/R_LLS', size(R_LLS), 'DataType', 'double');
h5write(h5File, '/R_LLS', R_LLS);

h5create(h5File, '/Q', size(Q), 'DataType', 'double');
h5write(h5File, '/Q', Q);

h5create(h5File, '/P0', size(P0), 'DataType', 'double');
h5write(h5File, '/P0', P0);

h5create(h5File, '/processNoise', size(processNoise), 'DataType', 'double');
h5write(h5File, '/processNoise', processNoise);

h5create(h5File, '/toaNoise', size(toaNoise), 'DataType', 'double');
h5write(h5File, '/toaNoise', toaNoise);

h5create(h5File, '/processbias', size(processbias), 'DataType', 'double');
h5write(h5File, '/processbias', processbias);

% Optional datasets for IMM analysis (not required by existing loaders)
h5create(h5File, '/true_state', size(true_state), 'DataType', 'double');
h5write(h5File, '/true_state', true_state);

h5create(h5File, '/mode_history', size(mode_history), 'DataType', 'double');
h5write(h5File, '/mode_history', mode_history);

fprintf('IMM-compatible data saved to %s\n', h5File);
end

function [trueStates, modeHistory, isValid] = localGenerateOneIMMRun( ...
    numSteps, x0, dt, omegas, TPM, initialModeProb, Q4, anchorMin, anchorMax, proposalMaxTry)
trueStates = zeros(4, numSteps);
modeHistory = zeros(1, numSteps);
isValid = true;

xk = x0;
currentMode = localSampleDiscrete(initialModeProb);

% Start exactly from the requested initial point and keep it recorded.
xk(1:2) = [0; 0];
trueStates(:, 1) = xk;
modeHistory(1) = currentMode;

for k = 2:numSteps
    currentMode = localSampleDiscrete(TPM(currentMode, :).');
    modeHistory(k) = currentMode;

    omega = omegas(currentMode);
    if abs(omega) < 1e-8
        F = [1, 0, dt, 0; ...
             0, 1, 0, dt; ...
             0, 0, 1, 0; ...
             0, 0, 0, 1];
    else
        wd = omega * dt;
        s = sin(wd);
        c = cos(wd);
        F = [1, 0, s/omega, -(1-c)/omega; ...
             0, 1, (1-c)/omega, s/omega; ...
             0, 0, c, -s; ...
             0, 0, s, c];
    end

    [xk, ok] = localProposeInsideBounds(xk, F, Q4, anchorMin, anchorMax, proposalMaxTry);
    if ~ok
        isValid = false;
        return;
    end
    trueStates(:, k) = xk;
end
end

function [xNext, ok] = localProposeInsideBounds(x, F, Q4, lowerBound, upperBound, maxTry)
% Draw process noise until predicted state remains inside bounds.
for t = 1:maxTry
    wk = localMvnrnd4(Q4);
    % Shrink process noise as retries increase to improve acceptance.
    wk = wk * (0.92 ^ (t - 1));
    cand = F * x + wk;
    if localIsInsideBox(cand(1:2), lowerBound, upperBound)
        xNext = cand;
        ok = true;
        return;
    end
end
xNext = x;
ok = false;
end

function tf = localIsInsideBox(pos, lowerBound, upperBound)
tf = all(pos >= lowerBound) && all(pos <= upperBound);
end

function idx = localSampleDiscrete(prob)
prob = prob(:);
prob = max(prob, 0);
total = sum(prob);
if total <= 0
    prob = ones(size(prob)) / numel(prob);
else
    prob = prob / total;
end
cdf = cumsum(prob);
r = rand;
idx = find(r <= cdf, 1, 'first');
if isempty(idx)
    idx = numel(prob);
end
end

function x = localMvnrnd4(Sigma)
Sigma = 0.5 * (Sigma + Sigma');
L = localRobustChol(Sigma);
x = L * randn(size(Sigma, 1), 1);
end

function L = localRobustChol(S)
I = eye(size(S, 1));
[L, p] = chol(S, 'lower');
if p == 0
    return;
end

for k = 0:7
    jitter = 1e-12 * (10 ^ k);
    [L, p] = chol(S + jitter * I, 'lower');
    if p == 0
        return;
    end
end

[V, D] = eig(0.5 * (S + S'));
d = max(diag(D), 1e-12);
L = V * diag(sqrt(d));
end
