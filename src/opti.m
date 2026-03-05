%% Parallel Particle Filter, Kalman Filter Evaluation
% Uses parfor to parallelize across noise levels
% Each noise level runs independently with its own filter instance

% clear all;
% close all;
% clc;

% % Env = Env(1e5);
% % Env.preSimulate();

% % % load data
% load('../data/z.mat');
% load('../data/toaPos.mat');
% load('../data/R.mat');
% load('../data/ranging.mat');
%% test
RMSE_obj = RMSE();
% parameters
numParticles = 500;
numIterations = 1e3;
pfIterations = 1e3;
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
toaPosition = zeros(2, numPoints, numIterations, numNoise);
for countNoise = 1:numNoise
    for countIter = 1:numIterations
        for countPoint = 2:numPoints
            toaPosition(:, countPoint, countIter, countNoise) = toaPos(:,countIter,countPoint,countNoise);
        end
    end
end
toaRMSE = RMSE_obj.getRMSE(toaPosition);

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
    pf_estimatedPos = zeros(2, numPoints, pfIterations);
    
    pf = ParticleFilter(countNoise, numParticles);
    
    for countIter = 1:pfIterations
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
    
    % Compute RMSE for this noise level
    rmse_temp = 0;
    for countIter = 1:pfIterations
        for countPoint = 1:numPoints
            rmse_temp = rmse_temp + norm(pf_estimatedPos(:, countPoint, countIter) - [countPoint; countPoint]);
        end
    end
    pf_RMSE(countNoise) = rmse_temp / (numPoints * pfIterations);
    
    pf_data_all{countNoise} = pf_estimatedPos;
end

pf_time = toc;
fprintf('PF (parallel) completed in %.2f seconds\n', pf_time);

%% Particle Filter nonLinear
pf_nonlinear_data_all = cell(numNoise, 1);
pf_nonlinear_RMSE = zeros(numNoise, 1);
% pfopti_w_gamma = [0.6 0.6 0.4 0.2 0.2];

tic;

parfor countNoise = 1:numNoise
    pf_estimatedPos = zeros(2, numPoints, pfIterations);
    
    pf = ParticleFilter(countNoise, numParticles);
    
    for countIter = 1:pfIterations
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
    
    % Compute RMSE for this noise level
    rmse_temp = 0;
    for countIter = 1:pfIterations
        for countPoint = 1:numPoints
            rmse_temp = rmse_temp + norm(pf_estimatedPos(:, countPoint, countIter) - [countPoint; countPoint]);
        end
    end
    pf_nonlinear_RMSE(countNoise) = rmse_temp / (numPoints * pfIterations);
    
    pf_nonlinear_data_all{countNoise} = pf_estimatedPos;
end

pf_nonlinear_time = toc;
fprintf('PF NonLinear (parallel) completed in %.2f seconds\n', pf_nonlinear_time);

%% Particle Filter Parameter Optimization (Multi-Parameter Approach)
%  Simultaneously optimize multiple parameters (gamma, numParticles)
%  Grid search over all combinations

% Define parameter ranges
gamma_candidates = linspace(0.01, 1.0, 15);      % 15 gamma values
particles_candidates = [100, 200, 500, 1000];    % Different particle counts
num_gamma = length(gamma_candidates);
num_particles_options = length(particles_candidates);

% Storage: rmse_grid(numGamma, numParticles, numNoise)
rmse_grid = zeros(num_gamma, num_particles_options, numNoise);
best_params = cell(numNoise, 1);

fprintf('\n=== Multi-Parameter Optimization ===\n');
fprintf('Testing %d gamma values × %d particle counts for %d noise levels\n', ...
    num_gamma, num_particles_options, numNoise);
fprintf('Total configurations: %d\n\n', num_gamma * num_particles_options);

total_tic = tic;

% Outer loop: particle count
for p_idx = 1:num_particles_options
    current_particles = particles_candidates(p_idx);
    
    % Inner loop: gamma values (parallelized)
    tic_gamma = tic;
    fprintf('Testing numParticles=%d ... ', current_particles);
    
    parfor gamma_idx = 1:num_gamma
        current_gamma = gamma_candidates(gamma_idx);
        
        % Evaluate for all noise levels
        for countNoise = 1:numNoise
            pf_estimatedPos = zeros(2, numPoints, pfIterations);
            pf = ParticleFilter(countNoise, current_particles);
            
            for countIter = 1:pfIterations
                particles_prev = [];
                vel_prev = [];
                weights_curr = ones(current_particles, 1) / current_particles;
                
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
                        particles_pred = pf.predictParam(particles_prev, vel_prev, 1, countPoint, current_gamma);
                        weights_upd = pf.update(particles_pred, weights_curr, meas, H, Rmat);
                        est = pf.estimate(particles_pred, weights_upd);
                        [particles_res, weights_upd] = pf.resamplingEss(particles_pred, weights_upd);
                        vel_new = est*ones(1, current_particles) - particles_prev;
                        
                        particles_prev = particles_res;
                        vel_prev = vel_new;
                        weights_curr = weights_upd;
                        
                        pf_estimatedPos(:, countPoint, countIter) = est;
                    end
                end
            end
            
            % Compute RMSE
            rmse_temp = 0;
            for countIter = 1:pfIterations
                for countPoint = 1:numPoints
                    rmse_temp = rmse_temp + norm(pf_estimatedPos(:, countPoint, countIter) - [countPoint; countPoint]);
                end
            end
            rmse_grid(gamma_idx, p_idx, countNoise) = rmse_temp / (numPoints * pfIterations);
        end
    end
    
    elapsed = toc(tic_gamma);
    fprintf('completed in %.1f sec\n', elapsed);
end

% Find best parameters for each noise level
for countNoise = 1:numNoise
    rmse_per_noise = rmse_grid(:, :, countNoise);  % num_gamma × num_particles_options
    [best_rmse, best_idx] = min(rmse_per_noise(:));
    [gamma_idx, p_idx] = ind2sub(size(rmse_per_noise), best_idx);
    
    best_params{countNoise} = struct( ...
        'gamma', gamma_candidates(gamma_idx), ...
        'numParticles', particles_candidates(p_idx), ...
        'rmse', best_rmse, ...
        'gamma_idx', gamma_idx, ...
        'particles_idx', p_idx);
end

total_optim_time = toc(total_tic);
fprintf('\n✓ Multi-parameter optimization completed in %.1f minutes (%.0f sec)\n', ...
    total_optim_time/60, total_optim_time);

% Display results
fprintf('\n--- Best Parameters per Noise Level ---\n');
fprintf('Noise Level  |  Best Gamma  |  Best Particles  |  RMSE\n');
fprintf('%-12s | %-12s | %-16s | %s\n', '             ', '            ', '                ', '       ');
for i = 1:numNoise
    fprintf('%.1e    |    %.3f     |      %4d        | %.4f\n', ...
        noisevalue(i), best_params{i}.gamma, best_params{i}.numParticles, best_params{i}.rmse);
end

%% Final evaluation with optimized parameters
fprintf('\n=== Final Evaluation with Optimized Parameters ===\n');

pf_opti_data_all = cell(numNoise, 1);
pf_opti_RMSE = zeros(numNoise, 1);
pf_opti_particles = zeros(numNoise, 1);
pf_opti_gamma = zeros(numNoise, 1);

tic;

parfor countNoise = 1:numNoise
    % Get optimized parameters for this noise level
    opt_gamma = best_params{countNoise}.gamma;
    opt_particles = best_params{countNoise}.numParticles;
    
    pf_estimatedPos = zeros(2, numPoints, pfIterations);
    pf = ParticleFilter(countNoise, opt_particles);
    
    for countIter = 1:pfIterations
        particles_prev = [];
        vel_prev = [];
        weights_curr = ones(opt_particles, 1) / opt_particles;
        
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
                particles_pred = pf.predictParam(particles_prev, vel_prev, 1, countPoint, opt_gamma);
                weights_upd = pf.update(particles_pred, weights_curr, meas, H, Rmat);
                est = pf.estimate(particles_pred, weights_upd);
                [particles_res, weights_upd] = pf.resamplingEss(particles_pred, weights_upd);
                vel_new = est*ones(1, opt_particles) - particles_prev;
                
                particles_prev = particles_res;
                vel_prev = vel_new;
                weights_curr = weights_upd;
                
                pf_estimatedPos(:, countPoint, countIter) = est;
            end
        end
    end
    
    % Compute RMSE
    rmse_temp = 0;
    for countIter = 1:pfIterations
        for countPoint = 1:numPoints
            rmse_temp = rmse_temp + norm(pf_estimatedPos(:, countPoint, countIter) - [countPoint; countPoint]);
        end
    end
    pf_opti_RMSE(countNoise) = rmse_temp / (numPoints * pfIterations);
    pf_opti_particles(countNoise) = opt_particles;
    pf_opti_gamma(countNoise) = opt_gamma;
    pf_opti_data_all{countNoise} = pf_estimatedPos;
end

final_eval_time = toc;
fprintf('Final evaluation completed in %.2f seconds\n', final_eval_time);

%% Visualization of Optimization Results

figure('Position', [100 100 1400 900]);

% Plot 1: Heatmap for first noise level (best example)
subplot(2, 3, 1);
imagesc(squeeze(rmse_grid(:, :, 3)));  % Middle noise level (1.0)
colorbar;
set(gca, 'XTick', 1:num_particles_options, 'YTick', 1:3:num_gamma);
set(gca, 'XTickLabel', cellstr(num2str(particles_candidates')), ...
    'YTickLabel', cellstr(num2str(round(gamma_candidates(1:3:num_gamma)', 3))));
xlabel('Number of Particles');
ylabel('Gamma Value');
title(sprintf('RMSE Heatmap (Noise=1.0)'));

% Plot 2: Best RMSE for each noise level with parameters
subplot(2, 3, 2);
opt_rmse_values = zeros(numNoise, 1);
for i = 1:numNoise
    opt_rmse_values(i) = best_params{i}.rmse;
end
bars = bar(1:numNoise, opt_rmse_values, 'FaceColor', [0.2 0.6 0.9]);
hold on;
for i = 1:numNoise
    text(i, opt_rmse_values(i) + 0.02, ...
        sprintf('N=%d\nγ=%.2f', best_params{i}.numParticles, best_params{i}.gamma), ...
        'HorizontalAlignment', 'center', 'FontSize', 8, 'FontWeight', 'bold');
end
set(gca, 'XTickLabel', cellstr(num2str(noisevalue)));
xlabel('Noise Level');
ylabel('RMSE');
grid on;
title('Optimized RMSE per Noise Level');

% Plot 3: Effect of number of particles (averaged across gammas)
subplot(2, 3, 3);
avg_rmse_per_particles = zeros(num_particles_options, 1);
for p_idx = 1:num_particles_options
    avg_rmse_per_particles(p_idx) = mean(mean(rmse_grid(:, p_idx, :)));
end
plot(particles_candidates, avg_rmse_per_particles, 'o-', 'LineWidth', 2, 'MarkerSize', 8);
xlabel('Number of Particles');
ylabel('Average RMSE (all gammas & noise)');
grid on;
title('Effect of Particle Count');

% Plot 4: Effect of gamma (averaged across particle counts)
subplot(2, 3, 4);
avg_rmse_per_gamma = zeros(num_gamma, 1);
for g_idx = 1:num_gamma
    avg_rmse_per_gamma(g_idx) = mean(mean(rmse_grid(g_idx, :, :)));
end
plot(gamma_candidates, avg_rmse_per_gamma, 's-', 'LineWidth', 2, 'MarkerSize', 6, 'Color', [0.8 0.2 0.2]);
xlabel('Gamma Value');
ylabel('Average RMSE (all particles & noise)');
grid on;
title('Effect of Gamma Parameter');

% Plot 5: RMSE heatmap for each noise level (3D view)
subplot(2, 3, 5);
best_per_noise = zeros(1, numNoise);
for i = 1:numNoise
    best_per_noise(i) = best_params{i}.rmse;
end
colors = [toaRMSE, pf_RMSE, pf_nonlinear_RMSE, best_per_noise'];
bar(1:numNoise, colors);
legend('ToA', 'PF (baseline)', 'PF (nonlinear)', 'PF (optimized)', 'Location', 'best');
set(gca, 'XTickLabel', cellstr(num2str(noisevalue)));
xlabel('Noise Level');
ylabel('RMSE');
grid on;
title('Filter Comparison - Optimized');

% Plot 6: Comparison of all PF methods
subplot(2, 3, 6);
semilogx(noisevalue, toaRMSE, 'o-', 'LineWidth', 2.5, 'MarkerSize', 8, 'DisplayName', 'ToA');
hold on;
semilogx(noisevalue, pf_RMSE, 's-', 'LineWidth', 2.5, 'MarkerSize', 8, 'DisplayName', 'PF (baseline)');
semilogx(noisevalue, pf_nonlinear_RMSE, '^-', 'LineWidth', 2.5, 'MarkerSize', 8, 'DisplayName', 'PF NonLinear');
semilogx(noisevalue, best_per_noise, 'd-', 'LineWidth', 2.5, 'MarkerSize', 8, 'DisplayName', 'PF Optimized');
xlabel('Noise Variance (log scale)');
ylabel('RMSE');
grid on;
legend('Location', 'best', 'FontSize', 10);
title('All Methods Comparison');

sgtitle('Multi-Parameter Particle Filter Optimization Analysis', 'FontSize', 14, 'FontWeight', 'bold');

%% Summary Report
fprintf('\n========================================\n');
fprintf('  MULTI-PARAMETER OPTIMIZATION REPORT   \n');
fprintf('========================================\n\n');

fprintf('Optimization Configuration:\n');
fprintf('  Gamma range:           [%.3f - %.3f] (%d values)\n', gamma_candidates(1), gamma_candidates(end), num_gamma);
fprintf('  Particles tested:      %s\n', sprintf('%d ', particles_candidates));
fprintf('  Total configurations:  %d\n', num_gamma * num_particles_options);
fprintf('  Noise levels:          %d\n', numNoise);
fprintf('  PF iterations per run: %d\n\n', pfIterations);

fprintf('--- Optimized Parameters by Noise Level ---\n');
fprintf('Noise Lvl | Gamma | Particles | RMSE    | Imp.vsBL(%%)\n');
fprintf('--------- | ----- | --------- | ------- | -----------\n');
for i = 1:numNoise
    improvement = (pf_RMSE(i) - best_params{i}.rmse) / pf_RMSE(i) * 100;
    fprintf('%.1e | %.3f | %9d | %.4f   | %+7.2f\n', ...
        noisevalue(i), best_params{i}.gamma, best_params{i}.numParticles, best_params{i}.rmse, improvement);
end

fprintf('\n--- Complete RMSE Comparison ---\n');
fprintf('Noise Lvl |   ToA   |   PF    | PF-NL   | PF-Opt  |\n');
fprintf('--------- | ------- | ------- | ------- | ------- |\n');
for i = 1:numNoise
    fprintf('%.1e | %.4f  | %.4f  | %.4f  | %.4f  |\n', ...
        noisevalue(i), toaRMSE(i), pf_RMSE(i), pf_nonlinear_RMSE(i), best_params{i}.rmse);
end

fprintf('\n--- Execution Times ---\n');
fprintf('Baseline PF (linear):        %.2f sec\n', pf_time);
fprintf('Baseline PF (non-linear):    %.2f sec\n', pf_nonlinear_time);
fprintf('Multi-parameter optimization: %.1f min (%.0f sec)\n', total_optim_time/60, total_optim_time);
fprintf('Final evaluation:            %.2f sec\n', final_eval_time);
fprintf('%-40s ------ \n', ' ');
fprintf('Overall optimization time:   %.1f min\n', (total_optim_time + final_eval_time)/60);

fprintf('\n========================================\n');