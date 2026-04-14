classdef EKFParticleFilter < handle
    properties
        NumParticles
        StateDim

        Particles
        Weights
        Covariances

        % System and observation model handles
        f_func
        F_func
        h_func
        H_func

        % Noise covariance matrices
        Q
        R

        % Numerical regularization for matrix factorization
        RegJitter = 1e-9
    end

    methods
        % Constructor
        function obj = EKFParticleFilter(numParticles, initialState, initialCov, f, F, h, H, Q, R)
            obj.NumParticles = numParticles;
            obj.StateDim = size(initialState, 1);

            obj.f_func = f;
            obj.F_func = F;
            obj.h_func = h;
            obj.H_func = H;
            obj.Q = Q;
            obj.R = R;

            obj.Weights = ones(1, numParticles) / numParticles;
            obj.Particles = zeros(obj.StateDim, numParticles);
            obj.Covariances = zeros(obj.StateDim, obj.StateDim, numParticles);

            L0 = obj.robustCholesky(initialCov);
            for i = 1:numParticles
                obj.Particles(:, i) = initialState + L0 * randn(obj.StateDim, 1);
                obj.Covariances(:, :, i) = initialCov;
            end
        end

        % Main predict-update step using EKF proposal
        function [estState, estCov] = step(obj, z_k)
            newParticles = zeros(obj.StateDim, obj.NumParticles);

            for i = 1:obj.NumParticles
                x_prev = obj.Particles(:, i);
                P_prev = obj.Covariances(:, :, i);

                % 1) EKF prediction
                x_pred = obj.f_func(x_prev);
                F_k = obj.F_func(x_prev);
                P_pred = F_k * P_prev * F_k' + obj.Q;
                P_pred = 0.5 * (P_pred + P_pred');

                % 2) EKF update for proposal parameters
                H_k = obj.H_func(x_pred);
                z_pred = obj.h_func(x_pred);

                S_k = H_k * P_pred * H_k' + obj.R;
                S_k = 0.5 * (S_k + S_k') + obj.RegJitter * eye(size(S_k, 1));
                K_k = (P_pred * H_k') / S_k;

                mu_q = x_pred + K_k * (z_k - z_pred);
                P_q = (eye(obj.StateDim) - K_k * H_k) * P_pred;
                P_q = 0.5 * (P_q + P_q') + obj.RegJitter * eye(obj.StateDim);

                % 3) Sample from proposal q(x_k | x_{k-1}, z_k)
                x_new = mu_q + obj.robustCholesky(P_q) * randn(obj.StateDim, 1);
                newParticles(:, i) = x_new;
                obj.Covariances(:, :, i) = P_q;

                % 4) Importance weight update
                innov = z_k - obj.h_func(x_new);
                p_zx = obj.gaussianPdf(innov, obj.R);

                state_diff = x_new - x_pred;
                p_xx = obj.gaussianPdf(state_diff, obj.Q);

                q_diff = x_new - mu_q;
                q_x = obj.gaussianPdf(q_diff, P_q);

                obj.Weights(i) = obj.Weights(i) * (p_zx * p_xx / (q_x + 1e-12));
            end

            wsum = sum(obj.Weights);
            if ~(isfinite(wsum) && wsum > 0)
                obj.Weights = ones(1, obj.NumParticles) / obj.NumParticles;
            else
                obj.Weights = obj.Weights / wsum;
            end

            obj.Particles = newParticles;

            % 5) Resample and return estimated state
            obj.resample();
            [estState, estCov] = obj.getStateEstimate();
        end

        % Systematic resampling with ESS threshold
        function resample(obj)
            ESS = 1 / sum(obj.Weights .^ 2);
            if ESS < obj.NumParticles / 2
                edges = min([0, cumsum(obj.Weights)], 1);
                edges(end) = 1;

                u1 = rand / obj.NumParticles;
                u = u1 + (0:(obj.NumParticles - 1)) / obj.NumParticles;
                [~, idx] = histc(u, edges);
                idx(idx < 1) = 1;

                obj.Particles = obj.Particles(:, idx);
                obj.Covariances = obj.Covariances(:, :, idx);
                obj.Weights = ones(1, obj.NumParticles) / obj.NumParticles;
            end
        end

        % Weighted mean and covariance estimate
        function [estState, estCov] = getStateEstimate(obj)
            estState = sum(obj.Particles .* obj.Weights, 2);

            estCov = zeros(obj.StateDim, obj.StateDim);
            for i = 1:obj.NumParticles
                err = obj.Particles(:, i) - estState;
                estCov = estCov + obj.Weights(i) * (err * err');
            end
        end

        function p = gaussianPdf(obj, diff, Sigma)
            n = size(diff, 1);
            L = obj.robustCholesky(Sigma);
            y = L \ diff;
            maha = y' * y;
            logDet = 2 * sum(log(abs(diag(L))));
            logp = -0.5 * (maha + logDet + n * log(2 * pi));
            p = exp(logp);
        end

        function L = robustCholesky(obj, S)
            Ssym = 0.5 * (S + S');
            I = eye(size(Ssym, 1));

            [L, p] = chol(Ssym, 'lower');
            if p == 0
                return;
            end

            for k = 0:8
                jitter = max(obj.RegJitter, 1e-12) * (10 ^ k);
                [L, p] = chol(Ssym + jitter * I, 'lower');
                if p == 0
                    return;
                end
            end

            [V, D] = eig(Ssym);
            d = max(diag(D), 1e-12);
            L = V * diag(sqrt(d));
        end
    end
end
