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

omegas = config.immOmegas(:).';
TPM = config.immTPM;
initialModeProb = config.immInitialModeProb(:);
x0 = config.immInitialState(:);

dt = config.immDt;
sigmaV = config.immSigmaV;
Q4 = [dt^3/3, 0, dt^2/2, 0; ...
      0, dt^3/3, 0, dt^2/2; ...
      dt^2/2, 0, dt, 0; ...
      0, dt^2/2, 0, dt] * sigmaV^2;

% Preallocate output tensors (same format as dataGenerate.m)
ranging_cell = cell(numNoises, 1);
x_hat_LLS_cell = cell(numNoises, 1);
z_LLS_cell = cell(numNoises, 1);
R_LLS_cell = cell(numNoises, 1);
processNoise_cell = cell(numNoises, 1);
toaNoise_cell = cell(numNoises, 1);
processbias_cell = cell(numNoises, 1);
Q_cell = cell(numNoises, 1);
P0_cell = cell(numNoises, 1);

% Optional debug/analysis tensors
true_state_cell = cell(numNoises, 1);
mode_history_cell = cell(numNoises, 1);

parfor n = 1:numNoises
    noiseVar = noiseVariance(n);

    ranging_temp = zeros(4, numPoints, numSamples);
    z_LLS_temp = zeros(6, numPoints, numSamples);
    R_LLS_temp = zeros(6, 6, numPoints, numSamples);
    x_hat_LLS_temp = zeros(2, numPoints, numSamples);

    processNoise_temp = zeros(2, numSamples);
    toaNoise_temp = zeros(2, numSamples);

    true_state_temp = zeros(4, numPoints, numSamples);
    mode_history_temp = zeros(numPoints, numSamples);

    for s = 1:numSamples
        [trueStates, modeHistory] = localGenerateOneIMMRun(numPoints, x0, dt, omegas, TPM, initialModeProb, Q4);

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

        % Keep process-noise bank/bias compatible with existing PF code.
        vel12 = x_hat_LLS_temp(:, 2, s) - x_hat_LLS_temp(:, 1, s);
        processNoise_temp(:, s) = trueStates(1:2, 3) - x_hat_LLS_temp(:, 2, s) - vel12;
        toaNoise_temp(:, s) = trueStates(1:2, 2) - x_hat_LLS_temp(:, 2, s);

        true_state_temp(:, :, s) = trueStates;
        mode_history_temp(:, s) = modeHistory(:);
    end

    processbias_temp = mean(processNoise_temp, 2);
    toabias_temp = mean(toaNoise_temp, 2);

    centeredProcess = processNoise_temp - processbias_temp;
    centeredToa = toaNoise_temp - toabias_temp;
    Q2 = (centeredProcess * centeredProcess.') / numSamples;
    P0 = (centeredToa * centeredToa.') / numSamples;

    ranging_cell{n} = ranging_temp;
    z_LLS_cell{n} = z_LLS_temp;
    R_LLS_cell{n} = R_LLS_temp;
    x_hat_LLS_cell{n} = x_hat_LLS_temp;

    processNoise_cell{n} = processNoise_temp;
    toaNoise_cell{n} = toaNoise_temp;
    processbias_cell{n} = processbias_temp;
    Q_cell{n} = Q2;
    P0_cell{n} = P0;

    true_state_cell{n} = true_state_temp;
    mode_history_cell{n} = mode_history_temp;
end

ranging = cat(4, ranging_cell{:});
x_hat_LLS = cat(4, x_hat_LLS_cell{:});
z_LLS = cat(4, z_LLS_cell{:});
R_LLS = cat(5, R_LLS_cell{:});

processNoise = cat(3, processNoise_cell{:});
toaNoise = cat(3, toaNoise_cell{:});
processbias = cat(2, processbias_cell{:});

Q = cat(3, Q_cell{:});
P0 = cat(3, P0_cell{:});

true_state = cat(4, true_state_cell{:});
mode_history = cat(3, mode_history_cell{:});

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

function [trueStates, modeHistory] = localGenerateOneIMMRun(numSteps, x0, dt, omegas, TPM, initialModeProb, Q4)
trueStates = zeros(4, numSteps);
modeHistory = zeros(1, numSteps);

xk = x0;
currentMode = localSampleDiscrete(initialModeProb);

for k = 1:numSteps
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

    wk = localMvnrnd4(Q4);
    xk = F * xk + wk;
    trueStates(:, k) = xk;
end
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
