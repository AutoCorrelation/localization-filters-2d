clear
clc
format long

% Add utils folder to path
addpath('./utils');
initializeParpool(5);
config = initializeConfig(1e3); % set particles
rng(42, 'twister'); % for reproducibility
% dataGenerate(config);
h5File = fullfile(config.pathData, 'simulation_data.h5');
data = loadSimulationData(h5File);

[lkf_pos, lkf_metric] = runFilter('LinearKalmanFilter', data, config);
[lpf_pos, lpf_metric] = runFilter('LinearParticleFilter', data, config);
[lkf_decayQ_pos, lkf_decayQ_metric] = runFilter('LinearKalmanFilter_DecayQ', data, config);
[nl_pf_pos, nl_pf_metric] = runFilter('NonlinearParticleFilter', data, config);
[baseline_pos, baseline_metric] = runFilter('Baseline', data, config);
% [customnlpf_pos, customnlpf_metric] = runFilter('CustomNonlinearParticleFilter', data, config);
[adaptivePF_pos, adaptivePF_metric] = runFilter('AdaptiveParticleFilter', data, config);
% 
[kldAdaptivePF_pos, kldAdaptivePF_metric] = runFilter('KLDAdaptiveParticleFilter', data, config);
[iaeMapAdaptivePF_pos, iaeMapAdaptivePF_metric] = runFilter('IAEMapAdaptiveParticleFilter', data, config);
[vbAdaptiveUPF_pos, vbAdaptiveUPF_metric] = runFilter('VariationalBayesianAdaptiveUPF', data, config);

% Display MAE results in table format
fprintf('\n=== MAE Comparison ===\n');
fprintf('Noise Level | baseline | LinearKF    | LinearPF    | LinearKF_DecayQ | NonLinearPF | AdaptivePF | KLDAdaptivePF | IAEMapAdaptivePF | VBAdaptiveUPF\n');
fprintf('-----------------------------------------------------------------------------------------------------------------------------------------------\n');
numNoise = numel(config.noiseVariance);
for i = 1:numNoise
    fprintf('%.0e   | %.4f | %.4f | %.4f | %.4f | %.4f | %.4f | %.4f | %.4f | %.4f\n', ...
        config.noiseVariance(i), baseline_metric.MAE(i), lkf_metric.MAE(i), lpf_metric.MAE(i), lkf_decayQ_metric.MAE(i), nl_pf_metric.MAE(i), adaptivePF_metric.MAE(i), kldAdaptivePF_metric.MAE(i), iaeMapAdaptivePF_metric.MAE(i), vbAdaptiveUPF_metric.MAE(i));
end
fprintf('\n');

plotMetricComparison(config.noiseVariance, baseline_metric.MAE, lkf_metric.MAE, lpf_metric.MAE, lkf_decayQ_metric.MAE, nl_pf_metric.MAE, adaptivePF_metric.MAE, kldAdaptivePF_metric.MAE, iaeMapAdaptivePF_metric.MAE, vbAdaptiveUPF_metric.MAE);