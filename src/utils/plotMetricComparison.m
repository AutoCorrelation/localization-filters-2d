function figHandle = plotMetricComparison(noiseVariance, apeMatrix, filterNames, runtimeTable, particleCount, resultDir, motionModel)
% plotMetricComparison - Plot APE and runtime summary in one figure with subplots.
% Usage:
%   plotMetricComparison(noiseVariance, apeMatrix, filterNames, runtimeTable, particleCount, resultDir, motionModel)
% motionModel: 'cv' or 'imm' for filename prefix

    if nargin < 7 || isempty(motionModel)
        motionModel = 'cv';
    end
    if nargin < 6 || isempty(resultDir)
        resultDir = [];
    end

filterLabels = localClassNamesToLegend(filterNames);

figHandle = figure('Name', sprintf('Benchmark (N=%d)', round(particleCount)), 'NumberTitle', 'off');

subplot(2, 1, 1);
hLines = semilogx(noiseVariance, apeMatrix, 'LineWidth', 1.5);

markerPool = {'o', 's', '^', 'v', 'd', 'p', 'h', 'x', '+', '*', '>'};
for i = 1:numel(hLines)
    hLines(i).Marker = markerPool{mod(i - 1, numel(markerPool)) + 1};
    hLines(i).MarkerSize = 7;
    hLines(i).MarkerIndices = 1:numel(noiseVariance);
end

legend(filterLabels, 'Location', 'northwest');
xlabel('Noise Variance');
ylabel('APE (Average Position Error)');
title(sprintf('APE Comparison by Noise Level (N=%d)', round(particleCount)));
grid on;

subplot(2, 1, 2);
runtimeMap = containers.Map(runtimeTable.FilterName, runtimeTable.RuntimeSec);
runtimeValues = zeros(1, numel(filterLabels));
for i = 1:numel(filterLabels)
    key = filterNames{i};
    if isKey(runtimeMap, key)
        runtimeValues(i) = runtimeMap(key);
    else
        runtimeValues(i) = NaN;
    end
end

bar(runtimeValues);
xticks(1:numel(filterLabels));
xticklabels(filterLabels);
xtickangle(35);
ylabel('Runtime (s)');
title(sprintf('Runtime Comparison (N=%d)', round(particleCount)));
grid on;

% Save figure if resultDir provided
if ~isempty(resultDir) && isdir(resultDir)
    motionPrefix = sprintf('%s_', motionModel);
    particleCountTag = sprintf('N%d', round(particleCount));
    figPath = fullfile(resultDir, sprintf('%s%s.fig', motionPrefix, particleCountTag));
    savefig(figHandle, figPath);
end

end

function labels = localClassNamesToLegend(filterNames)
    labels = filterNames;
    for i = 1:numel(filterNames)
        switch filterNames{i}
            case 'LinearKalmanFilter_DecayQ'
                labels{i} = 'LinearKF_DecayQ';
            case 'NonlinearParticleFilter'
                labels{i} = 'NonLinearPF';
            case 'EKFParticleFilter'
                labels{i} = 'EKF-PF';
            case 'AdaptiveParticleFilter'
                labels{i} = 'AdaptivePF(AdaBelief)';
            case 'BeliefQShrinkAdaptiveParticleFilter'
                labels{i} = 'BeliefQShrinkAdaptivePF';
            case 'RDiagPriorEditAdaptiveParticleFilter'
                labels{i} = 'RDiagPriorEditAdaptivePF';
            case 'BeliefRougheningAdaptiveParticleFilter'
                labels{i} = 'BeliefRougheningAdaptivePF';
            case 'RougheningPriorEditingParticleFilter'
                labels{i} = 'RougheningPriorEditingPF';
        end
    end
end
