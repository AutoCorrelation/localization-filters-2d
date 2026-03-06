clear
clc
format long

% Add utils folder to path
addpath('./utils');
initializeParpool(5);
config = initializeConfig();
dataGenerate(config);
h5File = fullfile(config.pathData, 'simulation_data.h5');
[ranging, x_hat_LLS, z_LLS, R_LLS, Q, P0, processNoise, toaNoise, processbias] = loadSimulationData(h5File);

