clear
clc
format long

% Add utils folder to path
addpath('./utils');
initializeParpool(5);
config = initializeConfig();
% dataGenerate(config);
h5File = fullfile(config.pathData, 'simulation_data.h5');
data = loadSimulationData(h5File);

[lkf_pos, lkf_RMSE] = runFilter('LinearKalmanFilter', data, config);
[lpf_pos, lpf_RMSE] = runFilter('LinearParticleFilter', data, config);
[lkf_decayQ_pos, lkf_decayQ_RMSE] = runFilter('LinearKalmanFilter_DecayQ', data, config);
[nl_pf_pos, nl_pf_RMSE] = runFilter('NonlinearParticleFilter', data, config);

% Display RMSE results in table format
fprintf('\n=== RMSE Comparison ===\n');
fprintf('Noise Level | LKF    | LPF    | LKF_DecayQ | NLPF\n');
fprintf('-----------------------------------------------------\n');
numNoise = numel(config.noiseVariance);
for i = 1:numNoise
    fprintf('%.0e   | %.4f | %.4f | %.4f     | %.4f\n', ...
        config.noiseVariance(i), lkf_RMSE(i), lpf_RMSE(i), lkf_decayQ_RMSE(i), nl_pf_RMSE(i));
end
fprintf('\n');