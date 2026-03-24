classdef AdaptiveParticleFilter < NonlinearParticleFilter
    % AdaptiveParticleFilter - AdaBelief + Adaptive R-inflation Particle Filter
    %
    % [Mathematical Background]
    %   Standard PF uses fixed R = sigma^2 * I measurement noise covariance.
    %   Diagonal inflation of R mitigates sample impoverishment while helping
    %   robustness and asymptotic optimality.
    %
    %   At each step, track global innovation with AdaBelief moments and
    %   inflate measurement covariance as: R_k = R_nom + lambdaR * diag(s_k).
    %
    %     e_k    = z_k - H(x_hat_k)                           (innovation/residual)
    %     m_k(i) = beta*m_{k-1}(i) + (1-beta)*e_k(i)         (1st moment, EMA)
    %     s_k(i) = beta*s_{k-1}(i) + (1-beta)*(e_k(i)-m_k(i))^2  (AdaBelief 2nd moment)
    %     R_k    = R_nom + lambdaR * diag(s_k)              (adaptive inflation)
    %
    %   Then weight update uses adaptive R_k:
    %     w_i = w_{i-1} * exp(-0.5 * e_i^T * R_k^{-1} * e_i)
    %
    % [Notes] beta close to 1: strong historical inertia (slow adaptation),
    %         beta close to 0: sensitive to recent innovations (fast adaptation).
    %
    % [Usage]
    %   filterObj = AdaptiveParticleFilter(data, config, noiseIdx);             % beta=0.99, lambdaR=1.0
    %   filterObj = AdaptiveParticleFilter(data, config, noiseIdx, 0.8, 2.0);  % beta=0.8, lambdaR=2.0

    properties
        % AdaBelief decay factor (EMA forgetting rate): 0 < beta < 1
        beta    (1,1) double = 0.99

        % R inflation strength: R_k = R_nom + lambdaR * diag(s_k)
        lambdaR (1,1) double = 1.0

        % R diagonal constraints (floor/ceiling for numerical stability)
        rFloor  (1,1) double = 1e-6
        rCeil   (1,1) double = 1e4
    end

    methods
        function obj = AdaptiveParticleFilter(data, config, noiseIdx, beta, lambdaR)
            % Constructor
            %   beta (optional): AdaBelief decay coefficient, default 0.99
            %   lambdaR (optional): R inflation strength, default 1.0
            obj@NonlinearParticleFilter(data, config, noiseIdx);

            if nargin >= 4 && ~isempty(beta)
                obj.beta = beta;
            end
            if nargin >= 5 && ~isempty(lambdaR)
                obj.lambdaR = lambdaR;
            end
        end

        function state = initializeState(obj, numPoints)
            % Initialize parent state + adaptive R-related fields
            state = initializeState@NonlinearParticleFilter(obj, numPoints);

            % Initialize nominal R diagonal and AdaBelief moments (1st, 2nd)
            numAnchors = size(obj.anchorPos, 1);
            nominalVar = obj.noiseScale^2;                    % noiseVariance(noiseIdx)
            state.nominalDiagR = nominalVar * ones(numAnchors, 1);
            state.mMoment = zeros(numAnchors, 1);
            state.sMoment = zeros(numAnchors, 1);
            state.diagR = state.nominalDiagR;
        end

        function [state, est] = step(obj, state, iterIdx, pointIdx)
            % -----------------------------------------------------------
            % 1. State prediction (from parent)
            % -----------------------------------------------------------
            particlesPred = state.particlesPrev + state.velPrev + obj.processBias + obj.sampleProcess();

            % -----------------------------------------------------------
            % 2. Current measurement
            % -----------------------------------------------------------
            zNow = obj.z(:, pointIdx, iterIdx);

            % -----------------------------------------------------------
            % 3. Compute global measurement innovation via weighted particle estimate
            %      x_hat_k = Σ w_i * x_i
            %      y_hat_k = H(x_hat_k), e_k = z_k - y_hat_k
            % -----------------------------------------------------------
            xHatWeighted  = particlesPred * state.weights;            % 2 x 1
            yPredWeighted = obj.H_nonlinear(xHatWeighted);            % numAnchors x 1
            e = zNow - yPredWeighted;                                  % numAnchors x 1

            % -----------------------------------------------------------
            % 4. AdaBelief belief moments update
            %      m(i) = beta*m(i) + (1-beta)*e(i)
            %      s(i) = beta*s(i) + (1-beta)*(e(i)-m(i))^2
            % -----------------------------------------------------------
            state.mMoment = obj.beta * state.mMoment + (1 - obj.beta) * e;
            state.sMoment = obj.beta * state.sMoment + (1 - obj.beta) * ((e - state.mMoment) .^ 2);

            % R inflation: R_k = R_nom + lambdaR * diag(s_k)
            state.diagR = state.nominalDiagR + obj.lambdaR * state.sMoment;

            % Apply floor/ceiling clips for numerical stability
            state.diagR = min(max(state.diagR, obj.rFloor), obj.rCeil);

            % -----------------------------------------------------------
            % 5. Update particle weights with adaptive R_k
            % -----------------------------------------------------------
            Rmat = diag(state.diagR);
            weightsUpd = obj.updateWeightsWithR(particlesPred, state.weights, zNow, Rmat);

            % -----------------------------------------------------------
            % 6. Particle state estimate and resampling
            % -----------------------------------------------------------
            est = particlesPred * weightsUpd;
            [particlesRes, weightsRes, idxResampled, didResample] = obj.resampleEssWithIndices(particlesPred, weightsUpd);

            if didResample
                state.velPrev = particlesRes - state.particlesPrev(:, idxResampled);
            else
                state.velPrev = particlesRes - state.particlesPrev;
            end
            state.particlesPrev = particlesRes;
            state.weights       = weightsRes;
            state.estimatedPos(:, pointIdx) = est;
        end

        function weights = updateWeightsWithR(obj, particles, prevWeights, zNow, Rmat)
            % Update particle importance weights using adaptive measurement covariance R
            %   p(z|x_i) = exp(-0.5 * e_i^T * R^{-1} * e_i)
            yPred  = obj.H_nonlinear(particles);     % numAnchors x numParticles
            errors = zNow - yPred;                   % numAnchors x numParticles
            Rinv   = diag(1 ./ diag(Rmat));          % Diagonal inverse of R (efficient for diagonal structure)

            distances = sum((Rinv * errors) .* errors, 1);  % 1 × numParticles

            weights = prevWeights(:)' .* exp(-0.5 * distances);
            weights = weights + 1e-300;
            weights = (weights / sum(weights)).';
        end
    end
end
