clear;
clc;
close all;

% plotAdaptiveTrajectory
% Visualize true trajectory, Adaptive PF estimated trajectory, and particles.

%% User settings
particleCount = 500;
noiseIdx = 1;        % 1..numel(config.noiseVariance)
sampleIdx = 1;       % 1..size(data.x_hat_LLS, 3)
showEveryKPoints = 1; % Particle scatter stride over trajectory points
saveFigure = false;
if evalin('base', "exist('SAVE_TRAJ_FIG','var')")
    saveFigure = evalin('base', 'logical(SAVE_TRAJ_FIG)');
end

%% Paths and setup
srcDir = fileparts(mfilename('fullpath'));
projectRoot = fileparts(srcDir);

addpath(fullfile(srcDir, 'utils'));
addpath(fullfile(srcDir, 'Filters'));

config = initializeConfig(particleCount);
% dataGenerate(config);
if strcmpi(config.motionModel, 'imm')
    h5FileName = 'simulation_data_imm.h5';
else
    h5FileName = 'simulation_data.h5';
end
h5File = fullfile(projectRoot, 'data', h5FileName);

if ~isfile(h5File)
    error('plotAdaptiveTrajectory:MissingData', 'Data file not found: %s', h5File);
end

%% Load data and validate indices
data = loadSimulationData(h5File);

numNoise = numel(config.noiseVariance);
numPoints = size(data.x_hat_LLS, 2);
numSamples = size(data.x_hat_LLS, 3);

if noiseIdx < 1 || noiseIdx > numNoise
    error('plotAdaptiveTrajectory:InvalidNoiseIdx', ...
        'noiseIdx must be in [1, %d], got %d', numNoise, noiseIdx);
end
if sampleIdx < 1 || sampleIdx > numSamples
    error('plotAdaptiveTrajectory:InvalidSampleIdx', ...
        'sampleIdx must be in [1, %d], got %d', numSamples, sampleIdx);
end

%% Build filter and run one sample trajectory
[bestBeta, bestLambdaR] = getBestParams(noiseIdx);
filterObj = AdaptiveParticleFilter(data, config, noiseIdx, bestBeta, bestLambdaR);

state = filterObj.initializeState(numPoints);
[state, p1, p2] = filterObj.initializeFirstTwo(state, sampleIdx);

estPos = zeros(2, numPoints);
estPos(:, 1) = p1;
estPos(:, 2) = p2;

particleHistory = cell(1, numPoints);
particleHistory{1} = filterObj.sampleToa(p1);
particleHistory{2} = state.particlesPrev;

for pointIdx = 3:numPoints
    [state, est] = filterObj.step(state, sampleIdx, pointIdx);
    estPos(:, pointIdx) = est;
    particleHistory{pointIdx} = state.particlesPrev;
end

truePos = data.true_state(1:2, 1:numPoints, sampleIdx, noiseIdx);
truePos = reshape(truePos, [2, numPoints]);

%% Prepare particle cloud for overview
idxPoints = 1:showEveryKPoints:numPoints;
allParticles = zeros(2, numel(idxPoints) * config.numParticles);
allPointTag = zeros(1, numel(idxPoints) * config.numParticles);

writeStart = 1;
for ii = 1:numel(idxPoints)
    pt = idxPoints(ii);
    p = particleHistory{pt};
    writeEnd = writeStart + size(p, 2) - 1;
    allParticles(:, writeStart:writeEnd) = p;
    allPointTag(writeStart:writeEnd) = pt;
    writeStart = writeEnd + 1;
end

%% Plot
figure('Color', 'w', 'Name', 'Adaptive PF Trajectory and Particles', 'Position', [80, 80, 1300, 560]);
tiledlayout(1, 2, 'TileSpacing', 'compact', 'Padding', 'compact');

% Left: trajectory + full particle cloud
nexttile;
hold on;
scatter(allParticles(1, :), allParticles(2, :), 7, allPointTag, 'filled', ...
    'MarkerFaceAlpha', 0.08, 'MarkerEdgeAlpha', 0.08);
plot(truePos(1, :), truePos(2, :), '-o', 'LineWidth', 2.4, 'Color', [0.00, 0.45, 0.74], ...
    'MarkerSize', 5, 'DisplayName', 'True Position');
plot(estPos(1, :), estPos(2, :), '-s', 'LineWidth', 2.1, 'Color', [0.85, 0.33, 0.10], ...
    'MarkerSize', 5, 'DisplayName', 'Adaptive PF Estimate');
scatter(config.Anchor(1, :), config.Anchor(2, :), 110, '^', 'filled', ...
    'MarkerFaceColor', [0.2, 0.2, 0.2], 'DisplayName', 'Anchors');
axis equal;
grid on;
xlabel('x');
ylabel('y');
colorbarHandle = colorbar;
ylabel(colorbarHandle, 'Trajectory Point Index');
title(sprintf('Noise Var = %.3g, Sample = %d, N = %d', ...
    config.noiseVariance(noiseIdx), sampleIdx, config.numParticles));
legend('Location', 'best');

% Right: particle snapshots at selected points
nexttile;
hold on;
ptsShow = unique([2, max(2, round(numPoints/2)), numPoints]);
palette = lines(numel(ptsShow));

for i = 1:numel(ptsShow)
    pt = ptsShow(i);
    p = particleHistory{pt};
    scatter(p(1, :), p(2, :), 10, 'filled', ...
        'MarkerFaceColor', palette(i, :), ...
        'MarkerFaceAlpha', 0.22, ...
        'MarkerEdgeAlpha', 0.22, ...
        'DisplayName', sprintf('Particles @ point %d', pt));

    plot(truePos(1, pt), truePos(2, pt), 'o', ...
        'MarkerSize', 8, ...
        'MarkerFaceColor', palette(i, :), ...
        'MarkerEdgeColor', 'k', ...
        'DisplayName', sprintf('True @ point %d', pt));

    plot(estPos(1, pt), estPos(2, pt), 's', ...
        'MarkerSize', 7, ...
        'MarkerFaceColor', palette(i, :), ...
        'MarkerEdgeColor', [0.15, 0.15, 0.15], ...
        'DisplayName', sprintf('Estimate @ point %d', pt));
end

scatter(config.Anchor(1, :), config.Anchor(2, :), 95, '^', 'filled', ...
    'MarkerFaceColor', [0.2, 0.2, 0.2], 'DisplayName', 'Anchors');
axis equal;
grid on;
xlabel('x');
ylabel('y');
title('Particle snapshots at selected points');
legend('Location', 'eastoutside');

if saveFigure
    outDir = fullfile(projectRoot, 'result');
    if ~exist(outDir, 'dir')
        mkdir(outDir);
    end
    outName = sprintf('adaptivepf_traj_noise%d_sample%d_N%d.png', noiseIdx, sampleIdx, config.numParticles);
    exportgraphics(gcf, fullfile(outDir, outName), 'Resolution', 200);
end
