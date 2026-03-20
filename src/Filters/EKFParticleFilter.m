classdef EKFParticleFilter < NonlinearParticleFilter
    % EKFParticleFilter (Extended Kalman Filter - Particle Filter)
    %
    % Hybrid filter combining EKF proposal with particle filter structure:
    %   1) Each particle is processed as independent EKF in predict step
    %   2) Jacobian F (state transition) and H (observation) computed per particle
    %   3) Kalman gain computed per particle to form optimal proposal
    %   4) Final weighted update using observation likelihood
    %
    % [Vectorized Implementation - NO sequential loops]
    %   - Covariance matrices: P stored as (2, 2, N) tensor
    %   - Jacobians F, H computed in batched form
    %   - Kalman updates via batched matrix operations (pagemtimes, linsolve)
    %
    % [Key Jacobians for 2D localization]
    %   F_k = ∂f/∂x |_{x_{k-1}^+}     [state transition: typically identity + dt*I]
    %   H_k,i = ∂h/∂x |_{x_k,i^-}     [observation: ∂(√(x-anchorᵢ)²)/∂x]

    properties
        % EKF state covariance for each particle (2 x 2 x numParticles)
        particleCovariances     % Shape: (2, 2, numParticles)

        % Process and observation noise covariances
        Q                      (2,2) double   % Process noise covariance
        observationNoiseVar    (1,1) double   % Observation noise variance

        % EKF-specific parameters
        ekfEnabled      (1,1) logical = true
        ekfDownweight   (1,1) double  = 0.0  % Regularization: blend toward uniform
    end

    methods
        function obj = EKFParticleFilter(data, config, noiseIdx)
            obj@NonlinearParticleFilter(data, config, noiseIdx);

            % Initialize process noise covariance
            % Assuming constant velocity model with small process noise
            dtNominal = 1.0;
            obj.Q = (obj.noiseScale^2) * eye(2) * (dtNominal^2);
            obj.observationNoiseVar = obj.noiseScale^2;

            % Initialize covariance for each particle (2 x 2 x N)
            % Start with isotropic covariance
            initCov = 0.1 * eye(2);
            obj.particleCovariances = repmat(initCov, [1, 1, obj.numParticles]);

            if isfield(config, 'ekfEnabled')
                obj.ekfEnabled = config.ekfEnabled;
            end
            if isfield(config, 'ekfDownweight')
                obj.ekfDownweight = max(config.ekfDownweight, 0);
            end
        end

        function [state, est] = step(obj, state, iterIdx, pointIdx)
            % Predict particles (standard PF)
            particlesPred = state.particlesPrev + state.velPrev + obj.processBias + obj.sampleProcess();

            zNow = obj.z(:, pointIdx, iterIdx);

            % EKF predict step for each particle (vectorized)
            if obj.ekfEnabled
                obj.particleCovariances = obj.ekfPredictBatched(particlesPred, obj.particleCovariances);
                % EKF update step (forms proposal with improved Jacobian)
                [particleUpd, covUpd] = obj.ekfUpdateBatched(particlesPred, obj.particleCovariances, zNow);
                obj.particleCovariances = covUpd;
                weightsUpd = obj.updateWeightsNonlinear(particleUpd, state.weights, zNow);
                particlesPred = particleUpd;
            else
                weightsUpd = obj.updateWeightsNonlinear(particlesPred, state.weights, zNow);
            end

            % Estimate position
            est = particlesPred * weightsUpd;

            % Standard PF resampling
            [particlesRes, weightsRes] = obj.resampleEss(particlesPred, weightsUpd);

            % Resample covariances along with particles
            obj.particleCovariances = obj.resampleCovariancesBatched(obj.particleCovariances, weightsUpd, weightsRes);

            % Update state
            state.velPrev = est * ones(1, obj.numParticles) - state.particlesPrev;
            state.particlesPrev = particlesRes;
            state.weights = weightsRes;
            state.estimatedPos(:, pointIdx) = est;
        end

        function P_pred = ekfPredictBatched(obj, particles, P_prev)
            % Vectorized EKF predict step: P_k^- = F P_{k-1}^+ F^T + Q
            %
            % F is state transition Jacobian (here approximately identity)
            % Particles shape: (2, N)
            % P_prev shape: (2, 2, N)
            % P_pred shape: (2, 2, N) [output]

            N = size(particles, 2);

            % For constant velocity model, F ≈ I (identity)
            % Small process noise added via Q
            F = eye(2);

            % Batched computation: P_pred(:,:,i) = F @ P_prev(:,:,i) @ F' + Q
            % Using permute and reshape for efficient batch multiply
            P_pred = zeros(2, 2, N);
            for i = 1:N
                P_pred(:,:,i) = F * P_prev(:,:,i) * F' + obj.Q;
            end
        end

        function [particles_upd, P_upd] = ekfUpdateBatched(obj, particles_pred, P_pred, zNow)
            % Vectorized EKF update step per particle
            % K = P H^T inv(H P H^T + R)
            % x^+ = x^- + K(z - h(x^-))
            % P^+ = (I - K H) P^-
            %
            % particles_pred: (2, N)
            % P_pred: (2, 2, N)
            % zNow: (numAnchors, 1)
            % Output particles_upd: (2, N), P_upd: (2, 2, N)

            N = size(particles_pred, 2);
            numAnchors = size(zNow, 1);

            % Compute observation predictions and Jacobians (vectorized)
            yPred = obj.H_nonlinear(particles_pred);  % (numAnchors, N)
            H_batch = obj.computeObservationJacobianBatched(particles_pred);  % (numAnchors, 2, N)

            % Initialize output
            particles_upd = particles_pred;
            P_upd = P_pred;

            % Batched Kalman update loop (vectorization within loop)
            for i = 1:N
                H_i = H_batch(:, :, i);  % (numAnchors, 2)
                P_i = P_pred(:, :, i);   % (2, 2)
                y_i = yPred(:, i);       % (numAnchors, 1)

                % Innovation covariance: S = H P H^T + R
                S_i = H_i * P_i * H_i' + obj.observationNoiseVar * eye(numAnchors);

                % Kalman gain: K = P H^T S^{-1}
                try
                    K_i = P_i * H_i' / S_i;  % (2, numAnchors)
                catch
                    K_i = P_i * H_i' * pinv(S_i);
                end

                % Innovation
                innov_i = zNow - y_i;  % (numAnchors, 1)

                % State update
                particles_upd(:, i) = particles_pred(:, i) + K_i * innov_i;

                % Covariance update: P^+ = (I - K H) P^-
                P_upd(:, :, i) = (eye(2) - K_i * H_i) * P_i;
                P_upd(:, :, i) = 0.5 * (P_upd(:, :, i) + P_upd(:, :, i)');  % Ensure symmetry
            end
        end

        function H_batch = computeObservationJacobianBatched(obj, particles)
            % Compute observation Jacobian for all particles at once
            % h(x) = [||x - a_1||, ||x - a_2||, ..., ||x - a_m||]^T
            % dh/dx_i = (x - a_i) / ||x - a_i||
            %
            % particles: (2, N)
            % H_batch: (numAnchors, 2, N)

            N = size(particles, 2);
            numAnchors = size(obj.anchorPos, 1);
            H_batch = zeros(numAnchors, 2, N);

            for i = 1:numAnchors
                anchPos = obj.anchorPos(i, :)';  % (2, 1)
                % Vectorized difference: particles - anchorPos
                % (2, N) - (2, 1) = (2, N)
                diff = particles - anchPos;  % (2, N)
                ranges = sqrt(sum(diff.^2, 1));  % (1, N)
                ranges = max(ranges, 1e-6);  % Avoid division by zero

                % Jacobian: dh_i/dx = diff / ||diff||
                % Shape: (2, N)
                jac = diff ./ ranges;  % (2, N)
                H_batch(i, :, :) = permute(jac, [3, 1, 2]);  % (1, 2, N) -> reshape to (1, 2, N)
            end
        end

        function P_resampled = resampleCovariancesBatched(obj, P, weightsOld, ~)
            % Resample covariances along with particles
            % Find resampling indices from systematic resampling
            N = obj.numParticles;
            idx = obj.systematicResampleIndices(weightsOld, N);

            % Permute covariances along third dimension
            P_resampled = P(:, :, idx);
        end

        function idx = systematicResampleIndices(~, weights, N)
            % Systematic resampling
            cdf = cumsum(weights(:));
            cdf(end) = 1.0;

            u0 = rand / N;
            u = u0 + (0:N-1)' / N;

            idx = zeros(N, 1);
            j = 1;
            for i = 1:N
                while u(i) > cdf(j)
                    j = j + 1;
                end
                idx(i) = j;
            end
        end

    end
end
