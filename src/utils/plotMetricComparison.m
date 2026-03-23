function figHandle = plotMetricComparison(noiseVariance, baseline_Metric, lkf_decayQ_Metric, nl_pf_Metric, ekf_Metric, adaptivePF_Metric, beliefQShrinkAdaptivePF_Metric, rDiagPriorEditAdaptivePF_Metric, beliefRougheningAdaptivePF_Metric, runtimeTable, particleCount)
% plotMetricComparison - Plot APE and runtime summary in one figure with subplots.
% Usage:
%   plotMetricComparison(noiseVariance, baseline_Metric, lkf_decayQ_Metric, nl_pf_Metric, ekf_Metric, adaptivePF_Metric, beliefQShrinkAdaptivePF_Metric, rDiagPriorEditAdaptivePF_Metric, beliefRougheningAdaptivePF_Metric, runtimeTable, particleCount)

filterLabels = {'Baseline', 'LinearKF_DecayQ', 'NonLinearPF', 'EKF-PF', 'AdaptivePF(AdaBelief)', 'BeliefQShrinkAdaptivePF', 'RDiagPriorEditAdaptivePF', 'BeliefRougheningAdaptivePF'};

figHandle = figure('Name', sprintf('Benchmark (N=%d)', round(particleCount)), 'NumberTitle', 'off');

subplot(2, 1, 1);
h = semilogx(noiseVariance, baseline_Metric, '-o', ...
              noiseVariance, lkf_decayQ_Metric, '-d', ...
              noiseVariance, nl_pf_Metric, '-x', ...
              noiseVariance, ekf_Metric, '-o', ...
              noiseVariance, adaptivePF_Metric, '-h', ...
              noiseVariance, beliefQShrinkAdaptivePF_Metric, '-.', ...
              noiseVariance, rDiagPriorEditAdaptivePF_Metric, '--', ...
              noiseVariance, beliefRougheningAdaptivePF_Metric, ':');

legend(filterLabels, 'Location', 'northwest');
xlabel('Noise Variance');
ylabel('APE (Average Position Error)');
title(sprintf('APE Comparison by Noise Level (N=%d)', round(particleCount)));
grid on;
set(h, 'LineWidth', 1.5);

subplot(2, 1, 2);
runtimeMap = containers.Map(runtimeTable.FilterName, runtimeTable.RuntimeSec);
runtimeValues = zeros(1, numel(filterLabels));
for i = 1:numel(filterLabels)
    key = localLegendToClassName(filterLabels{i});
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

end

function className = localLegendToClassName(label)
    switch label
        case 'Baseline'
            className = 'Baseline';
        case 'LinearKF_DecayQ'
            className = 'LinearKalmanFilter_DecayQ';
        case 'NonLinearPF'
            className = 'NonlinearParticleFilter';
        case 'EKF-PF'
            className = 'EKFParticleFilter';
        case 'AdaptivePF(AdaBelief)'
            className = 'AdaptiveParticleFilter';
        case 'BeliefQShrinkAdaptivePF'
            className = 'BeliefQShrinkAdaptiveParticleFilter';
        case 'RDiagPriorEditAdaptivePF'
            className = 'RDiagPriorEditAdaptiveParticleFilter';
        case 'BeliefRougheningAdaptivePF'
            className = 'BeliefRougheningAdaptiveParticleFilter';
        otherwise
            className = label;
    end
end
