clear
clc
format long

% Add utils folder to path
addpath('./utils');
initializeParpool(5);
basic = initializeConfig;
% dataGenerate(basic);
particleCounts = [10, 50, 100, 200, 500, 1000, 2000];

projectRoot = fileparts(fileparts(mfilename('fullpath')));
resultDir = fullfile(projectRoot, 'result');
if ~exist(resultDir, 'dir')
    mkdir(resultDir);
end

allMaeTable = table();

filterNames = {
    'Baseline', ...
    'LinearKalmanFilter', ...
    'LinearKalmanFilter_DecayQ', ...
    'LinearParticleFilter', ...
    'NonlinearParticleFilter', ...
    'RegularizedParticleFilter', ...
    'MCMCResamplingParticleFilter', ...
    'RougheningPriorEditingParticleFilter', ...
    'EKFParticleFilter', ...
    'AdaptiveParticleFilter', ...
    'ResidualSquaredAdaptiveParticleFilter', ...
    'BeliefQShrinkAdaptiveParticleFilter', ...
    'RDiagPriorEditAdaptiveParticleFilter', ...
    'BeliefRougheningAdaptiveParticleFilter', ...
    'KLDAdaptiveParticleFilter', ...
    'IAEMapAdaptiveParticleFilter'
};
filterClasses = filterNames;

maeVarNames = { ...
    'NoiseVariance', 'Baseline', 'LinearKalmanFilter', 'LinearKalmanFilter_DecayQ', ...
    'LinearParticleFilter', 'NonlinearParticleFilter', ...
    'RegularizedParticleFilter', 'MCMCResamplingParticleFilter', ...
    'RougheningPriorEditingParticleFilter', 'EKFParticleFilter', 'AdaptiveParticleFilter', ...
    'ResidualSquaredAdaptiveParticleFilter', ...
    'BeliefQShrinkAdaptiveParticleFilter', 'RDiagPriorEditAdaptiveParticleFilter', ...
    'BeliefRougheningAdaptiveParticleFilter', 'KLDAdaptiveParticleFilter', 'IAEMapAdaptiveParticleFilter'};

for nIdx = 1:numel(particleCounts)
    config = initializeConfig(particleCounts(nIdx));
    rng(42, 'twister'); % for reproducibility

    % dataGenerate(config);
    h5File = fullfile(config.pathData, 'simulation_data.h5');
    data = loadSimulationData(h5File);

    numFilters = numel(filterClasses);
    filterTimes = zeros(numFilters, 1);
    filterMetrics = cell(numFilters, 1);

    for fIdx = 1:numFilters
        tStart = tic;
        [~, metricOut] = runFilter(filterClasses{fIdx}, data, config);
        filterTimes(fIdx) = toc(tStart);
        filterMetrics{fIdx} = metricOut;
    end

    numNoise = numel(config.noiseVariance);
    maeMatrix = zeros(numNoise, numFilters);
    for fIdx = 1:numFilters
        maeMatrix(:, fIdx) = filterMetrics{fIdx}.MAE(:);
    end

    maeTable = array2table([config.noiseVariance(:), maeMatrix], ...
        'VariableNames', maeVarNames);

    runtimeTable = table(filterNames(:), filterTimes(:), ...
        'VariableNames', {'FilterName', 'RuntimeSec'});
    runtimeTable = sortrows(runtimeTable, 'RuntimeSec', 'ascend');

    fprintf('\n=== Particle Count: %d ===\n', round(config.numParticles));
    fprintf('=== Runtime Ranking (Fast -> Slow) ===\n');
    for i = 1:height(runtimeTable)
        fprintf('%2d. %-36s : %8.3f s\n', i, runtimeTable.FilterName{i}, runtimeTable.RuntimeSec(i));
    end
    fprintf('\n');

    maeSeries = num2cell(maeMatrix, 1);
    plotMetricComparison(config.noiseVariance, maeSeries{:}, runtimeTable, config.numParticles);

    savedPaths = saveBenchmarkResults(resultDir, config.numParticles, maeTable, runtimeTable);
    fprintf('Saved per-N files:\n');
    fprintf(' - %s\n', savedPaths.maeCsvPath);
    fprintf('\n');

    maeTableOut = addvars(savedPaths.maeTableWithRuntime, repmat(round(config.numParticles), height(savedPaths.maeTableWithRuntime), 1), ...
        'Before', 1, 'NewVariableNames', 'ParticleCount');
    allMaeTable = [allMaeTable; maeTableOut]; %#ok<AGROW>

end

allMaeCsvPath = fullfile(resultDir, sprintf('benchmark_batch_MAE_AllN.csv'));
writetable(allMaeTable, allMaeCsvPath);

fprintf('Saved aggregated files:\n');
fprintf(' - %s\n', allMaeCsvPath);
fprintf('\n');