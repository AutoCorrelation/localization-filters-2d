classdef KLDAdaptiveParticleFilter < NonlinearParticleFilter
    % KLDAdaptiveParticleFilter
    %
    % A-BPF / A-GPF style likelihood-adaptive particle filter.
    % At each time step, belief factor Theta is computed in closed-form:
    %   Theta_k = R_k * (H_k * Q_k * H_k' + R_k)^(-1)
    % and used to adapt likelihood with blended measurement:
    %   z_tilde = Theta_k * zPred + (I - Theta_k) * zNow
    %   p_AL(z|x_i) = p(z_tilde - h(x_i))

    properties
        thetaMin            (1,1) double = 0.0
        thetaMax            (1,1) double = 1.0
        regLambda           (1,1) double = 1e-8
        qFloor              (1,1) double = 1e-8
        qCeil               (1,1) double = 1e3
        thetaFallback       (1,1) double = 0.5
    end

    methods
        function obj = KLDAdaptiveParticleFilter(data, config, noiseIdx)
            obj@NonlinearParticleFilter(data, config, noiseIdx);

            if isfield(config, 'kldThetaMin')
                obj.thetaMin = config.kldThetaMin;
            end
            if isfield(config, 'kldThetaMax')
                obj.thetaMax = config.kldThetaMax;
            end
            if isfield(config, 'kldRegLambda')
                obj.regLambda = config.kldRegLambda;
            end
            if isfield(config, 'kldQFloor')
                obj.qFloor = config.kldQFloor;
            end
            if isfield(config, 'kldQCeil')
                obj.qCeil = config.kldQCeil;
            end
            if isfield(config, 'kldThetaFallback')
                obj.thetaFallback = config.kldThetaFallback;
            end
        end

        function [state, est] = step(obj, state, iterIdx, pointIdx)
            particlesPred = state.particlesPrev + state.velPrev + obj.processBias + obj.sampleProcess();

            zNow = obj.z(:, pointIdx, iterIdx);

            xPred = particlesPred * state.weights;
            zPred = obj.H_nonlinear(xPred);

            Qk = obj.estimatePredictionCovariance(particlesPred, state.weights);
            Hk = obj.computeJacobianAnalytical(xPred);
            numAnchors = size(Hk, 1);
            Rk = (obj.noiseScale ^ 2) * eye(numAnchors);

            Theta = obj.computeBeliefFactor(Hk, Qk, Rk);
            zBlend = Theta * zPred + (eye(numAnchors) - Theta) * zNow;

            weightsUpd = obj.updateWeightsAdaptiveLikelihood(particlesPred, state.weights, zBlend, Rk);

            est = particlesPred * weightsUpd;
            [particlesRes, weightsRes, idxResampled, didResample] = obj.resampleEssWithIndices(particlesPred, weightsUpd);

            if didResample
                state.velPrev = particlesRes - state.particlesPrev(:, idxResampled);
            else
                state.velPrev = particlesRes - state.particlesPrev;
            end
            state.particlesPrev = particlesRes;
            state.weights = weightsRes;
            state.estimatedPos(:, pointIdx) = est;
        end

        function H_jac = computeJacobianAnalytical(obj, x)
            numAnchors = size(obj.anchorPos, 1);
            H_jac = zeros(numAnchors, 2);

            for i = 1:numAnchors
                dx = x(1) - obj.anchorPos(i, 1);
                dy = x(2) - obj.anchorPos(i, 2);
                dist = sqrt(dx ^ 2 + dy ^ 2);

                if dist < 1e-10
                    dist = 1e-10;
                end

                H_jac(i, 1) = dx / dist;
                H_jac(i, 2) = dy / dist;
            end
        end

        function Qk = estimatePredictionCovariance(obj, particlesPred, weights)
            xMean = particlesPred * weights;
            centered = particlesPred - xMean;

            % Weighted covariance without forming an NxN diagonal matrix.
            weightedCentered = centered .* sqrt(weights(:)');
            QkRaw = weightedCentered * weightedCentered';
            QkRaw = 0.5 * (QkRaw + QkRaw');

            [V, D] = eig(QkRaw + obj.regLambda * eye(2));
            eigVals = diag(D);
            eigVals = min(max(eigVals, obj.qFloor), obj.qCeil);

            Qk = V * diag(eigVals) * V';
            Qk = 0.5 * (Qk + Qk');
        end

        function Theta = computeBeliefFactor(obj, Hk, Qk, Rk)
            innovationCov = Hk * Qk * Hk';
            denom = innovationCov + Rk + obj.regLambda * eye(size(Rk, 1));
            ThetaRaw = Rk / denom;

            if any(~isfinite(ThetaRaw), 'all')
                Theta = obj.thetaFallback * eye(size(Rk, 1));
                return;
            end

            ThetaSym = 0.5 * (ThetaRaw + ThetaRaw');
            [V, D] = eig(ThetaSym);
            eigVals = diag(D);
            eigVals = min(max(eigVals, obj.thetaMin), obj.thetaMax);

            Theta = V * diag(eigVals) * V';
            Theta = 0.5 * (Theta + Theta');
        end

        function weights = updateWeightsAdaptiveLikelihood(obj, particles, prevWeights, zBlend, Rk)
            yPred = obj.H_nonlinear(particles);
            errors = zBlend - yPred;

            Rstable = Rk + obj.regLambda * eye(size(Rk, 1));
            Rinv = Rstable \ eye(size(Rstable));
            distances = sum((Rinv * errors) .* errors, 1);

            weights = prevWeights(:)' .* exp(-0.5 * distances);
            weights = weights + 1e-300;
            weights = (weights / sum(weights)).';
        end
    end
end
