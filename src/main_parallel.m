%% Parallel Particle Filter, Kalman Filter Evaluation
% Uses parfor to parallelize across noise levels
% Each noise level runs independently with its own filter instance

clear all;
close all;
clc;

Env = Env(1e5);
Env.preSimulate();

% load data
load('../data/z.mat');
load('../data/toaPos.mat');
load('../data/R.mat');
load('../data/ranging.mat');
%% test
% parameters
numParticles = 0.5e3;
numIterations = 1e3;
numPoints = size(toaPos, 3);
numNoise = size(toaPos, 4);
H = [...
    0, -20
    20, -20
    20, 0
    20, 0
    20, 20
    0, 20];

%% TOA (sequential - no benefit from parallelization)
% toaPos is (2, numIterations, numPoints, numNoise); rearrange to
% (2, numPoints, numIterations, numNoise) via a single permute call.
toaPosition = permute(toaPos, [1, 3, 2, 4]);
toaRMSE = zeros(numNoise, 1);
for countNoise = 1:numNoise
    toaRMSE(countNoise) = rmsev(toaPosition(:, :, :, countNoise), 3);
end

%% Initialize parpool if not already running
poolobj = gcp('nocreate');
if isempty(poolobj)
    parpool(5);  % Use 5 workers; adjust based on your system
end


%% Particle Filter (parallelized across noise levels)
pf_data_all = cell(numNoise, 1);
pf_RMSE = zeros(numNoise, 1);
pfopti_w_gamma = [0.6 0.6 0.4 0.2 0.2];

tic;

parfor countNoise = 1:numNoise
    pf_estimatedPos = zeros(2, numPoints, numIterations);
    
    pf = ParticleFilter(countNoise, numParticles);
    
    for countIter = 1:numIterations
        particles_prev = [];
        vel_prev = [];
        weights_curr = ones(numParticles, 1) / numParticles;
        
        for countPoint = 2:numPoints
            meas = z(:, countIter, countPoint, countNoise);
            Rmat = R(:, :, countIter, countPoint, countNoise);

            if countPoint < 3
                pf_estimatedPos(:, countPoint-1, countIter) = toaPos(:, countIter, countPoint-1, countNoise);
                pf_estimatedPos(:, countPoint, countIter) = toaPos(:, countIter, countPoint, countNoise);

                p_prev = pf.sampling(toaPos(:, countIter, countPoint-1, countNoise));
                p_curr = pf.sampling(toaPos(:, countIter, countPoint, countNoise));
                particles_prev = p_curr;
                vel_prev = p_curr - p_prev;
            else
                % particles_pred = pf.predict(particles_prev, vel_prev, 1);
                particles_pred = pf.predictParam(particles_prev, vel_prev, 1, countPoint, pfopti_w_gamma(countNoise));
                weights_upd = pf.update(particles_pred, weights_curr, meas, H, Rmat);
                est = pf.estimate(particles_pred, weights_upd);
                [particles_res, weights_upd] = pf.resamplingEss(particles_pred, weights_upd);
                vel_new = est*ones(1, numParticles) - particles_prev;

                particles_prev = particles_res;
                vel_prev = vel_new;
                weights_curr = weights_upd;

                pf_estimatedPos(:, countPoint, countIter) = est;
            end
        end
    end
    
    pf_RMSE(countNoise) = rmsev(pf_estimatedPos, 3);
    
    pf_data_all{countNoise} = pf_estimatedPos;
end

pf_time = toc;
fprintf('PF (parallel) completed in %.2f seconds\n', pf_time);

%% Particle Filter nonLinear
pf_nonlinear_data_all = cell(numNoise, 1);
pf_nonlinear_RMSE = zeros(numNoise, 1);

tic;

parfor countNoise = 1:numNoise
    pf_estimatedPos = zeros(2, numPoints, numIterations);
    
    pf = ParticleFilter(countNoise, numParticles);
    
    for countIter = 1:numIterations
        particles_prev = [];
        vel_prev = [];
        weights_curr = ones(numParticles, 1) / numParticles;
        
        for countPoint = 2:numPoints
            if countPoint < 3
                pf_estimatedPos(:, countPoint-1, countIter) = toaPos(:, countIter, countPoint-1, countNoise);
                pf_estimatedPos(:, countPoint, countIter) = toaPos(:, countIter, countPoint, countNoise);

                p_prev = pf.sampling(toaPos(:, countIter, countPoint-1, countNoise));
                p_curr = pf.sampling(toaPos(:, countIter, countPoint, countNoise));
                particles_prev = p_curr;
                vel_prev = p_curr - p_prev;
            else
                particles_pred = pf.predict(particles_prev, vel_prev, 1);
                % particles_pred = pf.predictParam(particles_prev, vel_prev, 1, countPoint, pfopti_w_gamma(countNoise));
                weights_upd = pf.updateNonLinear(particles_pred, weights_curr, ranging(:, countIter, countPoint, countNoise));
                est = pf.estimate(particles_pred, weights_upd);
                [particles_res, weights_upd] = pf.resamplingEss(particles_pred, weights_upd);
                vel_new = est*ones(1, numParticles) - particles_prev;

                particles_prev = particles_res;
                vel_prev = vel_new;
                weights_curr = weights_upd;

                pf_estimatedPos(:, countPoint, countIter) = est;
            end
        end
    end
    
    pf_nonlinear_RMSE(countNoise) = rmsev(pf_estimatedPos, 3);
    
    pf_nonlinear_data_all{countNoise} = pf_estimatedPos;
end

pf_nonlinear_time = toc;
fprintf('PF NonLinear (parallel) completed in %.2f seconds\n', pf_nonlinear_time);

%% Kalman Filter (parallelized across noise levels)
kf_data_all = cell(numNoise, 1);
kf_RMSE = zeros(numNoise, 1);

tic;

parfor countNoise = 1:numNoise
    kf_data_noise = struct();
    kf_data_noise.estimatedPos = zeros(2, numPoints, numIterations);
    kf_data_noise.errCov = zeros(2, 2, numPoints, numIterations);
    kf_data_noise.vel = zeros(2, numPoints, numIterations);
    
    kf = KalmanFilter(countNoise, H);
    
    for countIter = 1:numIterations
        for countPoint = 2:numPoints
            if countPoint < 3
                kf_data_noise.estimatedPos(:, countPoint-1, countIter) = toaPos(:, countIter, countPoint-1, countNoise);
                kf_data_noise.estimatedPos(:, countPoint, countIter) = toaPos(:, countIter, countPoint, countNoise);
                kf_data_noise.vel(:, countPoint, countIter) = kf_data_noise.estimatedPos(:, countPoint, countIter) - kf_data_noise.estimatedPos(:, countPoint-1, countIter);
            else
                [xhat, Phat] = kf.predict(kf_data_noise.estimatedPos(:, countPoint-1, countIter), kf_data_noise.errCov(:, :, countPoint-1, countIter), kf_data_noise.vel(:, countPoint-1, countIter), 1);
                kf = kf.update(Phat, R(:, :, countIter, countPoint, countNoise));
                [kf_data_noise.estimatedPos(:, countPoint, countIter), kf_data_noise.errCov(:,:, countPoint, countIter)] = kf.estimate(xhat, Phat, z(:, countIter, countPoint, countNoise));
                kf_data_noise.vel(:, countPoint, countIter) = kf_data_noise.estimatedPos(:, countPoint, countIter) - kf_data_noise.estimatedPos(:, countPoint-1, countIter);
            end
        end
    end
    
    kf_RMSE(countNoise) = rmsev(kf_data_noise.estimatedPos, 3);
    
    kf_data_all{countNoise} = kf_data_noise;
end

kf_time = toc;
fprintf('KF (parallel) completed in %.2f seconds\n', kf_time);

%% Kalman Filter Modified (parallelized across noise levels)
kf1_data_all = cell(numNoise, 1);
kf1_RMSE = zeros(numNoise, 1);
optimal_gamma = [0.5, 0.5, 0.4, 0.3, 0.5];

tic;

parfor countNoise = 1:numNoise
    kf1_data_noise = struct();
    kf1_data_noise.estimatedPos = zeros(2, numPoints, numIterations);
    kf1_data_noise.errCov = zeros(2, 2, numPoints, numIterations);
    kf1_data_noise.vel = zeros(2, numPoints, numIterations);
    
    kf = KalmanFilter(countNoise, H);
    
    for countIter = 1:numIterations
        for countPoint = 2:numPoints
            if countPoint < 3
                kf1_data_noise.estimatedPos(:, countPoint-1, countIter) = toaPos(:, countIter, countPoint-1, countNoise);
                kf1_data_noise.estimatedPos(:, countPoint, countIter) = toaPos(:, countIter, countPoint, countNoise);
                kf1_data_noise.vel(:, countPoint, countIter) = kf1_data_noise.estimatedPos(:, countPoint, countIter) - kf1_data_noise.estimatedPos(:, countPoint-1, countIter);
            else
                [xhat, Phat] = kf.predict_decayQ(kf1_data_noise.estimatedPos(:, countPoint-1, countIter), kf1_data_noise.errCov(:, :, countPoint-1, countIter), kf1_data_noise.vel(:, countPoint-1, countIter), 1, countPoint, optimal_gamma(countNoise));
                kf = kf.update(Phat, R(:, :, countIter, countPoint, countNoise));
                [kf1_data_noise.estimatedPos(:, countPoint, countIter), kf1_data_noise.errCov(:,:, countPoint, countIter)] = kf.estimate(xhat, Phat, z(:, countIter, countPoint, countNoise));
                kf1_data_noise.vel(:, countPoint, countIter) = kf1_data_noise.estimatedPos(:, countPoint, countIter) - kf1_data_noise.estimatedPos(:, countPoint-1, countIter);
            end
        end
    end
    
    kf1_RMSE(countNoise) = rmsev(kf1_data_noise.estimatedPos, 3);
    
    kf1_data_all{countNoise} = kf1_data_noise;
end

kf1_time = toc;
fprintf('KF1 (parallel) completed in %.2f seconds\n', kf1_time);

%% Plotting
noisevalue = [0.01; 0.1; 1; 10; 100];

figure;
semilogx(noisevalue, kf_RMSE, 'DisplayName', 'Kalman Filter');
hold on;
semilogx(noisevalue, kf1_RMSE, 'DisplayName', 'Kalman Filter 1');
semilogx(noisevalue, pf_RMSE, 'DisplayName', 'Particle Filter', 'LineWidth', 1.5);
semilogx(noisevalue, toaRMSE, 'DisplayName', 'ToA');
semilogx(noisevalue, pf_nonlinear_RMSE, 'DisplayName', 'Particle Filter NonLinear', 'LineWidth', 1.5);
legend show;
grid on;
title('Parallel Execution - Filter Comparison');
xlabel('Noise variance');
ylabel('RMSE');

%% Summary
fprintf('\n=== Execution Times ===\n');
fprintf('PF (parallel):          %.2f sec\n', pf_time);
fprintf('PF NonLinear (parallel): %.2f sec\n', pf_nonlinear_time);
fprintf('KF (parallel):          %.2f sec\n', kf_time);
fprintf('KF1 (parallel):         %.2f sec\n', kf1_time);
fprintf('Total:                  %.2f sec\n', pf_time + pf_nonlinear_time + kf_time + kf1_time);

fprintf('\n=== RMSE Comparison ===\n');
fprintf('Noise Level | ToA    | KF     | KF1    | PF     | PF(NonLinear) \n');
fprintf('------------------------------------------------\n');
for i = 1:numNoise
    fprintf('%.0e   | %.4f | %.4f | %.4f | %.4f | %.4f\n', ...
        noisevalue(i), toaRMSE(i), kf_RMSE(i), kf1_RMSE(i), pf_RMSE(i), pf_nonlinear_RMSE(i));
end