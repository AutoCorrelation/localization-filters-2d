classdef AuxiliaryParticleFilter < NonlinearParticleFilter
    % AuxiliaryParticleFilter (APF)
    %
    % APF addresses particle collapse by look-ahead selection:
    %   1) Predict particles: x_k^- = x_{k-1}^+ + v_{k-1} + ...
    %   2) Compute predicted likelihood q_i ∝ p(y_k | μ_k,i)
    %   3) Resample based on q_i (auxiliary weights)
    %   4) Update final weights using actual likelihood p(y_k | x_k,i)
    %
    % [Look-ahead mechanism]
    %   q_i = w_{k-1,i} * p(y_k | μ_k,i)  where μ_k,i = E[x_k | x_k,i^-]
    %
    % [Nonlinearity-based branching]
    %   If system nonlinearity is weak, APF weight degradation is suppressed.
    %   Nonlinearity is estimated from residual variance; if weak, skip APF.
    %
    % [Optional probability smoothing]
    %   q_tilde = α*(α-1)*q_i + q_bar   (α ∈ [1, ∞))
    %   Mitigates extreme weight concentration when α > 1.

    properties
        % APF-specific parameters
        apfNonlinearityThreshold (1,1) double = 0.5  % Threshold for nonlinearity detection
        apfSmoothingAlpha        (1,1) double = 1.1  % α ∈ [1, ∞), 1 = no smoothing
        apfEnableLookAhead       (1,1) logical = true % Enable/disable apf look-ahead
    end

    methods
        function obj = AuxiliaryParticleFilter(data, config, noiseIdx)
            obj@NonlinearParticleFilter(data, config, noiseIdx);

            if isfield(config, 'apfNonlinearityThreshold')
                obj.apfNonlinearityThreshold = max(config.apfNonlinearityThreshold, 0);
            end
            if isfield(config, 'apfSmoothingAlpha')
                obj.apfSmoothingAlpha = max(config.apfSmoothingAlpha, 1.0);
            end
            if isfield(config, 'apfEnableLookAhead')
                obj.apfEnableLookAhead = config.apfEnableLookAhead;
            end
        end

        function [state, est] = step(obj, state, iterIdx, pointIdx)
            % Predict particles
            particlesPred = state.particlesPrev + state.velPrev + obj.processBias + obj.sampleProcess();

            zNow = obj.z(:, pointIdx, iterIdx);

            % APF Look-ahead: compute auxiliary weights q_i
            if obj.apfEnableLookAhead
                qAux = obj.computeAuxiliaryWeights(particlesPred, state.weights, zNow);
                [particlesRes, weightsAux] = obj.resampleEss(particlesPred, qAux);
            else
                % Fallback: standard resampling
                [particlesRes, weightsAux] = obj.resampleEss(particlesPred, state.weights);
            end

            % Final weight update using actual likelihood
            weightsUpd = obj.updateWeightsNonlinear(particlesRes, weightsAux, zNow);

            % Estimate position
            est = particlesRes * weightsUpd;

            % Update state
            state.velPrev = est * ones(1, obj.numParticles) - state.particlesPrev;
            state.particlesPrev = particlesRes;
            state.weights = weightsUpd;
            state.estimatedPos(:, pointIdx) = est;
        end

        function qAux = computeAuxiliaryWeights(obj, particlesPred, prevWeights, zNow)
            % Compute auxiliary weights based on predicted likelihood
            %   q_i = w_{k-1,i} * p(y_k | μ_k,i)
            %
            % μ_k,i approximation: weighted mean or particle mean
            % For simplicity, use predicted particle mean across all particles.

            % Measure nonlinearity from current observation residuals
            yPred = obj.H_nonlinear(particlesPred);
            residuals = zNow - yPred;
            residualVar = var(residuals, 1, 2);
            nonlinearityScore = mean(residualVar);

            % Branch based on nonlinearity strength
            if nonlinearityScore < obj.apfNonlinearityThreshold * (obj.noiseScale^2)
                % Weak nonlinearity: skip APF to avoid weight corruption
                qAux = prevWeights;
                return;
            end

            % Strong nonlinearity: compute particle-wise look-ahead weights
            % μ_k,i is approximated by each predicted particle state.
            yPredMu = obj.H_nonlinear(particlesPred);
            llhMu = obj.computeLogLikelihood(zNow, yPredMu);
            llhMu = llhMu - max(llhMu);

            % Auxiliary weights q_i ∝ w_{k-1,i} * p(y_k | μ_k,i)
            qAux = prevWeights(:)' .* exp(llhMu);

            % Optional: Apply probability smoothing
            if obj.apfSmoothingAlpha > 1.0
                qBar = ones(size(qAux)) / numel(qAux);  % Uniform reference
                alpha = obj.apfSmoothingAlpha;
                qAux = alpha * (alpha - 1) * qAux + (1 - alpha * (alpha - 1)) * qBar;
            end

            % Normalize
            qAux = max(qAux, 0);
            qAux = qAux + 1e-300;
            qSum = sum(qAux);
            if ~isfinite(qSum) || qSum <= 0
                qAux = prevWeights(:);
            else
                qAux = (qAux / qSum).';
            end
        end

        function llh = computeLogLikelihood(obj, zNow, yPred)
            % Compute Gaussian log-likelihood
            numAnchors = size(yPred, 1);
            Rinv = eye(numAnchors) / (obj.noiseScale^2);
            errors = zNow - yPred;
            distances = sum((Rinv * errors) .* errors, 1);
            llh = -0.5 * distances;
        end
    end
end
