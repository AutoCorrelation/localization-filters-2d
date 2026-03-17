clear
clc
format long

% Add utils folder to path
addpath('./utils');
initializeParpool(5);
config = initializeConfig(1e3);
% dataGenerate(config);
h5File = fullfile(config.pathData, 'simulation_data.h5');
data = loadSimulationData(h5File);

[lkf_pos, lkf_RMSE] = runFilter('LinearKalmanFilter', data, config);
[lpf_pos, lpf_RMSE] = runFilter('LinearParticleFilter', data, config);
[lkf_decayQ_pos, lkf_decayQ_RMSE] = runFilter('LinearKalmanFilter_DecayQ', data, config);
[nl_pf_pos, nl_pf_RMSE] = runFilter('NonlinearParticleFilter', data, config);
[baseline_pos, baseline_RMSE] = runFilter('Baseline', data, config);
% [customnlpf_pos, customnlpf_RMSE] = runFilter('CustomNonlinearParticleFilter', data, config);
[adaptivePF_pos, adaptivePF_RMSE] = runFilter('AdaptiveParticleFilter', data, config);

% Display RMSE results in table format
fprintf('\n=== RMSE Comparison ===\n');
fprintf('Noise Level | baseline | LinearKF    | LinearPF    | LinearKF_DecayQ | NonLinearPF | AdaptivePF\n');
fprintf('--------------------------------------------------------------------------\n');
numNoise = numel(config.noiseVariance);
for i = 1:numNoise
    fprintf('%.0e   | %.4f | %.4f | %.4f | %.4f | %.4f | %.4f\n', ...
        config.noiseVariance(i), baseline_RMSE(i), lkf_RMSE(i), lpf_RMSE(i), lkf_decayQ_RMSE(i), nl_pf_RMSE(i), adaptivePF_RMSE(i));
end
fprintf('\n');

plotRMSEComparison(config.noiseVariance, baseline_RMSE, lkf_RMSE, lpf_RMSE, lkf_decayQ_RMSE, nl_pf_RMSE, adaptivePF_RMSE);