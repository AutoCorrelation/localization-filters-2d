clear
clc
format long

% Add utils folder to path
addpath('./utils');
initializeParpool(5);
basic = initializeConfig;
% dataGenerate(basic);
particleCounts = [10, 50, 100, 200, 500, 1e3];

projectRoot = fileparts(fileparts(mfilename('fullpath')));
resultDir = fullfile(projectRoot, 'result');
if ~exist(resultDir, 'dir')
    mkdir(resultDir);
end

allRmseTable = table();

filterNames = {
    'Baseline';
    'LinearKalmanFilter_DecayQ';
    'NonlinearParticleFilter';
    'RegularizedParticleFilter';
    % 'EKFParticleFilter';
    'AdaptiveParticleFilter';
    'RDiagPriorEditAdaptiveParticleFilter';
    'RougheningPriorEditingParticleFilter';
    'RBPF'
};
filterClasses = filterNames;

rmseVarNames = [{'NoiseVariance'}; filterNames];

for nIdx = 1:numel(particleCounts)
    config = initializeConfig(particleCounts(nIdx));
    rng(42, 'twister'); % for reproducibility

    % Uncomment below to regenerate data with current config.motionModel setting
    % dataGenerate(config);
    
    % Select h5 file based on motion model
    if strcmpi(config.motionModel, 'imm')
        h5FileName = 'simulation_data_imm.h5';
    else
        h5FileName = 'simulation_data.h5';
    end
    h5File = fullfile(config.pathData, h5FileName);
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
    rmseMatrix = zeros(numNoise, numFilters);
    for fIdx = 1:numFilters
        rmseMatrix(:, fIdx) = filterMetrics{fIdx}.RMSE(:);
    end

    rmseTable = array2table([config.noiseVariance(:), rmseMatrix], ...
        'VariableNames', rmseVarNames);

    runtimeTable = table(filterNames(:), filterTimes(:), ...
        'VariableNames', {'FilterName', 'RuntimeSec'});
    runtimeTable = sortrows(runtimeTable, 'RuntimeSec', 'ascend');

    fprintf('\n=== Particle Count: %d ===\n', round(config.numParticles));
    fprintf('=== Runtime Ranking (Fast -> Slow) ===\n');
    for i = 1:height(runtimeTable)
        fprintf('%2d. %-36s : %8.3f s\n', i, runtimeTable.FilterName{i}, runtimeTable.RuntimeSec(i));
    end
    fprintf('\n');

    plotMetricComparison(config.noiseVariance, rmseMatrix, filterNames, runtimeTable, config.numParticles, resultDir, config.motionModel);

    savedPaths = saveBenchmarkResults(resultDir, config.numParticles, rmseTable, runtimeTable, config.motionModel);
    fprintf('Saved per-N files:\n');
    fprintf(' - %s\n', savedPaths.rmseCsvPath);
    fprintf('\n');

    rmseTableOut = addvars(savedPaths.rmseTableWithRuntime, repmat(round(config.numParticles), height(savedPaths.rmseTableWithRuntime), 1), ...
        'Before', 1, 'NewVariableNames', 'ParticleCount');
    allRmseTable = [allRmseTable; rmseTableOut]; %#ok<AGROW>

end

motionPrefix = sprintf('%s_', basic.motionModel);
allRmseCsvPath = fullfile(resultDir, sprintf('benchmark_%sbatch_RMSE_AllN.csv', motionPrefix));
writetable(allRmseTable, allRmseCsvPath);

fprintf('Saved aggregated files:\n');
fprintf(' - %s\n', allRmseCsvPath);
fprintf('\n');
