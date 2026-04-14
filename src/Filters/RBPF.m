classdef RBPF < NonlinearParticleFilter
    % RBPF - Rao-Blackwellized Particle Filter adapted to this codebase.
    %
    % Decomposition used here (minimal integration with existing pipeline):
    %   x^(1): position (nonlinear part, particle representation)
    %   x^(2): velocity (conditionally linear Gaussian, per-particle KF)
    %
    % Notes:
    % - Ranging likelihood is computed from nonlinear position particles.
    % - A per-particle KF tracks velocity using a linear pseudo-measurement
    %   derived from LLS position increments.

    properties
        q2Scale (1,1) double = 1.0
        velMeasScale (1,1) double = 1.0
        initVelCovScale (1,1) double = 1.0
        regJitter (1,1) double = 1e-9
    end

    methods
        function obj = RBPF(data, config, noiseIdx)
            obj@NonlinearParticleFilter(data, config, noiseIdx);

            if isfield(config, 'rbpfQ2Scale')
                obj.q2Scale = max(config.rbpfQ2Scale, 1e-6);
            end
            if isfield(config, 'rbpfVelMeasScale')
                obj.velMeasScale = max(config.rbpfVelMeasScale, 1e-6);
            end
            if isfield(config, 'rbpfInitVelCovScale')
                obj.initVelCovScale = max(config.rbpfInitVelCovScale, 1e-6);
            end
            if isfield(config, 'rbpfRegJitter')
                obj.regJitter = max(config.rbpfRegJitter, 1e-12);
            end
        end

        function state = initializeState(obj, numPoints)
            state = initializeState@LinearParticleFilter(obj, numPoints);

            velDim = 2;
            initCov = obj.initVelCovScale * (obj.noiseStd ^ 2) * eye(velDim);

            state.kfMean = zeros(velDim, obj.numParticles);
            state.kfCov = repmat(initCov, [1, 1, obj.numParticles]);
        end

        function [state, p1, p2] = initializeFirstTwo(obj, state, iterIdx)
            p1 = obj.xHat(:, 1, iterIdx);
            p2 = obj.xHat(:, 2, iterIdx);

            state.estimatedPos(:, 1) = p1;
            state.estimatedPos(:, 2) = p2;

            sampledPrev = obj.sampleToa(p1);
            sampledCurr = obj.sampleToa(p2);

            state.particlesPrev = sampledCurr;
            velInit = sampledCurr - sampledPrev;
            state.kfMean = velInit;
        end

        function [state, est] = step(obj, state, iterIdx, pointIdx)
            N = obj.numParticles;
            velDim = 2;
            numAnchors = size(obj.anchorPos, 1);

            q2 = obj.q2Scale * (obj.noiseStd ^ 2) * eye(velDim);
            rVel = obj.velMeasScale * (obj.noiseStd ^ 2) * eye(velDim);
            rRange = 0.5 * (obj.R + obj.R') + obj.regJitter * eye(numAnchors);
            I2 = eye(2);

            % Pseudo linear measurement for velocity from LLS position increment.
            zVel = obj.xHat(:, pointIdx, iterIdx) - obj.xHat(:, pointIdx - 1, iterIdx);

            particlesPred = zeros(2, N);
            kfMeanPred = zeros(velDim, N);
            kfCovPred = zeros(velDim, velDim, N);
            kfMeanUpd = zeros(velDim, N);
            kfCovUpd = zeros(velDim, velDim, N);
            logW = zeros(N, 1);

            logPrevW = log(max(state.weights(:), 1e-300));

            for i = 1:N
                mPrev = state.kfMean(:, i);
                PPrev = state.kfCov(:, :, i);

                % Linear dynamics for velocity (A = I by default in this model).
                mPred = mPrev;
                PPred = PPrev + q2;
                PPred = 0.5 * (PPred + PPred') + obj.regJitter * I2;

                % Sample velocity from per-particle conditional Gaussian.
                vSample = obj.sampleGaussian(mPred, PPred);

                % Nonlinear position propagation.
                particlesPred(:, i) = state.particlesPrev(:, i) + vSample + obj.processBias + obj.sampleProcessSingle();

                % KF update on velocity with pseudo linear measurement (H = I).
                Svel = PPred + rVel;
                Svel = 0.5 * (Svel + Svel') + obj.regJitter * I2;
                Kvel = PPred / Svel;

                innovVel = zVel - mPred;
                mUpd = mPred + Kvel * innovVel;
                PUpd = (I2 - Kvel) * PPred * (I2 - Kvel)' + Kvel * rVel * Kvel';
                PUpd = 0.5 * (PUpd + PUpd') + obj.regJitter * I2;

                kfMeanPred(:, i) = mPred;
                kfCovPred(:, :, i) = PPred;
                kfMeanUpd(:, i) = mUpd;
                kfCovUpd(:, :, i) = PUpd;

                % Marginal likelihood for ranging measurement.
                yPred = obj.H_nonlinear(particlesPred(:, i));
                J = obj.computeRangeJacobian(particlesPred(:, i));
                Srange = J * PPred * J' + rRange;
                Srange = 0.5 * (Srange + Srange') + obj.regJitter * eye(numAnchors);

                innovRange = obj.z(:, pointIdx, iterIdx) - yPred;
                logLike = obj.logGaussianZeroMean(innovRange, Srange);

                logW(i) = logPrevW(i) + logLike;
            end

            weightsUpd = obj.normalizeLogWeights(logW);
            est = particlesPred * weightsUpd;

            [particlesRes, weightsRes, idxResampled, didResample] = obj.resampleEssWithIndices(particlesPred, weightsUpd);
            if didResample
                state.kfMean = kfMeanUpd(:, idxResampled);
                state.kfCov = kfCovUpd(:, :, idxResampled);
            else
                state.kfMean = kfMeanUpd;
                state.kfCov = kfCovUpd;
            end

            state.particlesPrev = particlesRes;
            state.weights = weightsRes;
            state.estimatedPos(:, pointIdx) = est;
        end

        function J = computeRangeJacobian(obj, x)
            numAnchors = size(obj.anchorPos, 1);
            J = zeros(numAnchors, 2);

            for a = 1:numAnchors
                d = x - obj.anchorPos(a, :)';
                r = norm(d);
                if r < 1e-10
                    r = 1e-10;
                end
                J(a, :) = (d / r)';
            end
        end

        function noise = sampleProcessSingle(obj)
            if isempty(obj.processNoise)
                noise = obj.noiseStd * randn(2, 1);
                return;
            end

            idx = randi(size(obj.processNoise, 2));
            noise = obj.processNoise(:, idx);
        end

        function x = sampleGaussian(obj, mu, Sigma)
            L = obj.robustCholesky(Sigma);
            x = mu + L * randn(size(mu, 1), 1);
        end

        function logp = logGaussianZeroMean(obj, e, S)
            L = obj.robustCholesky(S);
            y = L \ e;
            maha = y' * y;
            logDet = 2 * sum(log(abs(diag(L))));
            d = size(e, 1);
            logp = -0.5 * (maha + logDet + d * log(2 * pi));
        end

        function w = normalizeLogWeights(~, logW)
            m = max(logW);
            shifted = logW - m;
            lse = m + log(sum(exp(shifted)));
            w = exp(logW - lse);
            w = w + 1e-300;
            w = w / sum(w);
        end

        function L = robustCholesky(obj, S)
            Ssym = 0.5 * (S + S');
            I = eye(size(Ssym, 1));

            [L, p] = chol(Ssym, 'lower');
            if p == 0
                return;
            end

            base = max(1e-12, obj.regJitter);
            for k = 0:7
                jitter = base * (10 ^ k);
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
