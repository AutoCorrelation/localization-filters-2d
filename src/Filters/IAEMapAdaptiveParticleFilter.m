classdef IAEMapAdaptiveParticleFilter < NonlinearParticleFilter
    % IAEMapAdaptiveParticleFilter
    % Innovation-based Adaptive Estimation (IAE) + MAP parameter estimation PF.
    %
    % This filter estimates latent model parameters(theta) while adapting Q_k online:
    %   C_z(k) = (1/L) * sum_{j=k-L+1}^{k} nu_j * nu_j'
    %   Q_k    = H_k^+ * (C_z(k) - R_k) * (H_k^+)'
    % where nu_k = z_k - h(x_hat_k), H_k^+ is pseudo-inverse Jacobian.

    properties
        windowLength        (1,1) double = 20
        qFloor              (1,1) double = 1e-6
        qCeil               (1,1) double = 10
        regLambda           (1,1) double = 1e-8

        parameterJitterStd  (1,1) double = 1e-3
        mapFeedbackGain     (1,1) double = 0.2
        processQScale       (1,1) double = 1.0
    end

    methods
        function obj = IAEMapAdaptiveParticleFilter(data, config, noiseIdx)
            obj@NonlinearParticleFilter(data, config, noiseIdx);

            if isfield(config, 'iaeWindowLength')
                obj.windowLength = max(2, round(config.iaeWindowLength));
            end
            if isfield(config, 'iaeQFloor')
                obj.qFloor = config.iaeQFloor;
            end
            if isfield(config, 'iaeQCeil')
                obj.qCeil = config.iaeQCeil;
            end
            if isfield(config, 'iaeRegLambda')
                obj.regLambda = config.iaeRegLambda;
            end
            if isfield(config, 'mapParameterJitterStd')
                obj.parameterJitterStd = config.mapParameterJitterStd;
            end
            if isfield(config, 'mapFeedbackGain')
                obj.mapFeedbackGain = config.mapFeedbackGain;
            end
            if isfield(config, 'iaeProcessQScale')
                obj.processQScale = config.iaeProcessQScale;
            end
        end

        function state = initializeState(obj, numPoints)
            state = initializeState@NonlinearParticleFilter(obj, numPoints);

            numAnchors = size(obj.anchorPos, 1);
            q0 = (obj.noiseScale ^ 2) * obj.processQScale * eye(2);

            state.Q = q0;
            state.thetaParticles = repmat(obj.processBias, 1, obj.numParticles) + ...
                obj.parameterJitterStd * randn(2, obj.numParticles);
            state.thetaMap = obj.processBias;

            state.residualBuffer = zeros(numAnchors, obj.windowLength);
            state.residualHead = 1;
            state.residualCount = 0;
        end

        function [state, est] = step(obj, state, iterIdx, pointIdx)
            processNoise = obj.sampleProcessFromQ(state.Q);
            thetaNow = state.thetaParticles;
            particlesPred = state.particlesPrev + state.velPrev + thetaNow + processNoise;

            zNow = obj.z(:, pointIdx, iterIdx);
            weightsUpd = obj.updateWeightsNonlinear(particlesPred, state.weights, zNow);
            est = particlesPred * weightsUpd;

            innovation = zNow - obj.H_nonlinear(est);
            state = obj.pushInnovation(state, innovation);

            Czk = obj.computeInnovationCovariance(state);
            Hk = obj.computeJacobianAnalytical(est);
            Rk = (obj.noiseScale ^ 2) * eye(size(Hk, 1));
            QkRaw = obj.inverseMapQ(Czk, Rk, Hk);
            state.Q = obj.projectCovariance(QkRaw, state.Q);

            [particlesRes, thetaRes, weightsRes, resampleIdx, didResample] = ...
                obj.resampleEssWithParameters(particlesPred, thetaNow, weightsUpd);

            state.thetaMap = obj.computeMapParameter(thetaNow, weightsUpd, resampleIdx, didResample);
            feedbackTarget = state.thetaMap * ones(1, obj.numParticles);
            state.thetaParticles = (1 - obj.mapFeedbackGain) * thetaRes + ...
                obj.mapFeedbackGain * feedbackTarget + ...
                obj.parameterJitterStd * randn(2, obj.numParticles);

            state.velPrev = est * ones(1, obj.numParticles) - state.particlesPrev;
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

        function noise = sampleProcessFromQ(obj, Q)
            Qsym = 0.5 * (Q + Q');
            Qsym = Qsym + obj.regLambda * eye(2);

            [V, D] = eig(Qsym);
            eigVals = diag(D);
            eigVals = max(eigVals, obj.qFloor);
            L = V * diag(sqrt(eigVals));

            noise = L * randn(2, obj.numParticles);
        end

        function state = pushInnovation(obj, state, innovation)
            state.residualBuffer(:, state.residualHead) = innovation;
            state.residualHead = state.residualHead + 1;
            if state.residualHead > obj.windowLength
                state.residualHead = 1;
            end

            state.residualCount = min(state.residualCount + 1, obj.windowLength);
        end

        function Czk = computeInnovationCovariance(obj, state)
            if state.residualCount < obj.windowLength
                residuals = state.residualBuffer(:, 1:state.residualCount);
            else
                residuals = state.residualBuffer;
            end

            if isempty(residuals)
                Czk = (obj.noiseScale ^ 2) * eye(size(state.residualBuffer, 1));
                return;
            end

            Czk = (residuals * residuals') / size(residuals, 2);
            Czk = 0.5 * (Czk + Czk');
        end

        function Qk = inverseMapQ(obj, Czk, Rk, Hk)
            innovationDriven = Czk - Rk;
            innovationDriven = 0.5 * (innovationDriven + innovationDriven');

            % Pseudo-inverse(SVD 기반)로 특이 Jacobian 상황 안정화.
            Hplus = pinv(Hk, obj.regLambda);
            Qk = Hplus * innovationDriven * Hplus';
            Qk = 0.5 * (Qk + Qk');
        end

        function Qproj = projectCovariance(obj, Qraw, Qfallback)
            if any(~isfinite(Qraw), 'all')
                Qproj = Qfallback;
                return;
            end

            [V, D] = eig(0.5 * (Qraw + Qraw'));
            eigVals = diag(D);
            eigVals = min(max(eigVals, obj.qFloor), obj.qCeil);
            Qproj = V * diag(eigVals) * V';
            Qproj = 0.5 * (Qproj + Qproj') + obj.regLambda * eye(2);
        end

        function [particlesOut, thetaOut, weightsOut, indices, didResample] = ...
                resampleEssWithParameters(obj, particles, thetaParticles, weights)
            N = obj.numParticles;
            ess = 1 / sum(weights .^ 2);

            if ess >= N / 2
                didResample = false;
                indices = (1:N)';
                particlesOut = particles;
                thetaOut = thetaParticles;
                weightsOut = weights;
                return;
            end

            didResample = true;
            cdf = cumsum(weights(:));
            u0 = rand / N;
            u = u0 + (0:N-1)' / N;

            indices = zeros(N, 1);
            j = 1;
            for i = 1:N
                while (j < N) && (u(i) > cdf(j))
                    j = j + 1;
                end
                indices(i) = j;
            end

            particlesOut = particles(:, indices);
            thetaOut = thetaParticles(:, indices);
            weightsOut = ones(N, 1) / N;
        end

        function thetaMap = computeMapParameter(obj, thetaPred, weights, resampleIdx, didResample)
            if didResample
                counts = accumarray(resampleIdx, 1, [obj.numParticles, 1]);
                [~, modeIdx] = max(counts);
                thetaMap = thetaPred(:, modeIdx);
                return;
            end

            [~, mapIdx] = max(weights);
            thetaMap = thetaPred(:, mapIdx);
        end
    end
end
