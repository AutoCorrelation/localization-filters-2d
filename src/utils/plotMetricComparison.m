function figHandle = plotMetricComparison(noiseVariance, baseline_Metric, lkf_Metric, lkf_decayQ_Metric, lpf_Metric, nl_pf_Metric, rpf_Metric, mcmc_Metric, apf_Metric, rougheningPF_Metric, ekf_Metric, adaptivePF_Metric, beliefQShrinkAdaptivePF_Metric, rDiagPriorEditAdaptivePF_Metric, beliefRougheningAdaptivePF_Metric, kldAdaptivePF_Metric, iaeMapAdaptivePF_Metric, runtimeTable, particleCount)
% plotMetricComparison - Plot MAE and runtime summary in one figure with subplots.
% Usage:
%   plotMetricComparison(noiseVariance, baseline_Metric, lkf_Metric, lkf_decayQ_Metric, lpf_Metric, nl_pf_Metric, rpf_Metric, mcmc_Metric, apf_Metric, rougheningPF_Metric, ekf_Metric, adaptivePF_Metric, beliefQShrinkAdaptivePF_Metric, rDiagPriorEditAdaptivePF_Metric, beliefRougheningAdaptivePF_Metric, kldAdaptivePF_Metric, iaeMapAdaptivePF_Metric, runtimeTable, particleCount)

filterLabels = {'Baseline', 'LinearKF', 'LinearKF_DecayQ', 'LinearPF', 'NonLinearPF', 'RPF', 'MCMC', 'APF', 'RougheningPF', 'EKF-PF', 'AdaptivePF', 'BeliefQShrinkAdaptivePF', 'RDiagPriorEditAdaptivePF', 'BeliefRougheningAdaptivePF', 'KLDAdaptivePF', 'IAEMapAdaptivePF'};

figHandle = figure('Name', sprintf('Benchmark (N=%d)', round(particleCount)), 'NumberTitle', 'off');

subplot(2, 1, 1);
h = semilogx(noiseVariance, baseline_Metric, '-o', ...
              noiseVariance, lkf_Metric, '-s', ...
              noiseVariance, lkf_decayQ_Metric, '-d', ...
              noiseVariance, lpf_Metric, '-^', ...
              noiseVariance, nl_pf_Metric, '-x', ...
              noiseVariance, rpf_Metric, '-+', ...
              noiseVariance, mcmc_Metric, '->', ...
              noiseVariance, apf_Metric, '-<', ...
              noiseVariance, rougheningPF_Metric, '-*', ...
              noiseVariance, ekf_Metric, '-o', ...
              noiseVariance, adaptivePF_Metric, '-h', ...
              noiseVariance, beliefQShrinkAdaptivePF_Metric, '-.', ...
              noiseVariance, rDiagPriorEditAdaptivePF_Metric, '--', ...
              noiseVariance, beliefRougheningAdaptivePF_Metric, ':', ...
              noiseVariance, kldAdaptivePF_Metric, '-p', ...
              noiseVariance, iaeMapAdaptivePF_Metric, '-v');

legend(filterLabels, 'Location', 'northwest');
xlabel('Noise Variance');
ylabel('MAE');
title(sprintf('MAE Comparison by Noise Level (N=%d)', round(particleCount)));
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
        case 'LinearKF'
            className = 'LinearKalmanFilter';
        case 'LinearKF_DecayQ'
            className = 'LinearKalmanFilter_DecayQ';
        case 'LinearPF'
            className = 'LinearParticleFilter';
        case 'NonLinearPF'
            className = 'NonlinearParticleFilter';
        case 'RPF'
            className = 'RegularizedParticleFilter';
        case 'MCMC'
            className = 'MCMCResamplingParticleFilter';
        case 'APF'
            className = 'AuxiliaryParticleFilter';
        case 'RougheningPF'
            className = 'RougheningPriorEditingParticleFilter';
        case 'EKF-PF'
            className = 'EKFParticleFilter';
        case 'AdaptivePF'
            className = 'AdaptiveParticleFilter';
        case 'BeliefQShrinkAdaptivePF'
            className = 'BeliefQShrinkAdaptiveParticleFilter';
        case 'RDiagPriorEditAdaptivePF'
            className = 'RDiagPriorEditAdaptiveParticleFilter';
        case 'BeliefRougheningAdaptivePF'
            className = 'BeliefRougheningAdaptiveParticleFilter';
        case 'KLDAdaptivePF'
            className = 'KLDAdaptiveParticleFilter';
        case 'IAEMapAdaptivePF'
            className = 'IAEMapAdaptiveParticleFilter';
        otherwise
            className = label;
    end
end
