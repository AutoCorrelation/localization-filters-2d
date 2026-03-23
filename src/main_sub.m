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

allApeTable = table();

filterNames = {
    'Baseline', ...
    'LinearKalmanFilter_DecayQ', ...
    'NonlinearParticleFilter', ...
    'EKFParticleFilter', ...
    'AdaptiveParticleFilter', ...
    'BeliefQShrinkAdaptiveParticleFilter', ...
    'RDiagPriorEditAdaptiveParticleFilter', ...
    'BeliefRougheningAdaptiveParticleFilter'
};
filterClasses = filterNames;

apeVarNames = { ...
    'NoiseVariance', 'Baseline', 'LinearKalmanFilter_DecayQ', ...
    'NonlinearParticleFilter', ...
    'EKFParticleFilter', ...
    'AdaptiveParticleFilter', ...
    'BeliefQShrinkAdaptiveParticleFilter', 'RDiagPriorEditAdaptiveParticleFilter', ...
    'BeliefRougheningAdaptiveParticleFilter'};

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
    apeMatrix = zeros(numNoise, numFilters);
    for fIdx = 1:numFilters
        apeMatrix(:, fIdx) = filterMetrics{fIdx}.APE(:);
    end

    apeTable = array2table([config.noiseVariance(:), apeMatrix], ...
        'VariableNames', apeVarNames);

    runtimeTable = table(filterNames(:), filterTimes(:), ...
        'VariableNames', {'FilterName', 'RuntimeSec'});
    runtimeTable = sortrows(runtimeTable, 'RuntimeSec', 'ascend');

    fprintf('\n=== Particle Count: %d ===\n', round(config.numParticles));
    fprintf('=== Runtime Ranking (Fast -> Slow) ===\n');
    for i = 1:height(runtimeTable)
        fprintf('%2d. %-36s : %8.3f s\n', i, runtimeTable.FilterName{i}, runtimeTable.RuntimeSec(i));
    end
    fprintf('\n');

    plotMetricComparison(config.noiseVariance, ...
        apeMatrix(:, 1), ...
        apeMatrix(:, 2), ...
        apeMatrix(:, 3), ...
        apeMatrix(:, 4), ...
        apeMatrix(:, 5), ...
        apeMatrix(:, 6), ...
        apeMatrix(:, 7), ...
        apeMatrix(:, 8), ...
        runtimeTable, config.numParticles);

    savedPaths = saveBenchmarkResults(resultDir, config.numParticles, apeTable, runtimeTable);
    fprintf('Saved per-N files:\n');
    fprintf(' - %s\n', savedPaths.apeCsvPath);
    fprintf('\n');

    apeTableOut = addvars(savedPaths.apeTableWithRuntime, repmat(round(config.numParticles), height(savedPaths.apeTableWithRuntime), 1), ...
        'Before', 1, 'NewVariableNames', 'ParticleCount');
    allApeTable = [allApeTable; apeTableOut]; %#ok<AGROW>

end

allApeCsvPath = fullfile(resultDir, sprintf('benchmark_batch_APE_AllN.csv'));
writetable(allApeTable, allApeCsvPath);

fprintf('Saved aggregated files:\n');
fprintf(' - %s\n', allApeCsvPath);
fprintf('\n');
