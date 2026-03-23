classdef EKFParticleFilter < NonlinearParticleFilter
    % EKFParticleFilter
    %
    % EKF-proposal particle filter:
    %   q(x_k|x_{k-1}, z_k) = N(mu_q, P_q) from per-particle EKF update
    %   w_k^i proportional to w_{k-1}^i * p(z_k|x_k^i) * p(x_k^i|x_{k-1}^i) / q(x_k^i|x_{k-1}^i, z_k)

    properties
        particleCovariances
        initCovariance (2,2) double = 0.1 * eye(2)

        Q (2,2) double
        regJitter (1,1) double = 1e-9

        ekfEnabled (1,1) logical = true
    end

    methods
        function obj = EKFParticleFilter(data, config, noiseIdx)
            obj@NonlinearParticleFilter(data, config, noiseIdx);

            qScale = 1.0;
            if isfield(config, 'ekfQScale')
                qScale = max(config.ekfQScale, 1e-6);
            end
            obj.Q = (obj.noiseScale ^ 2) * qScale * eye(2);

            if isfield(config, 'ekfUseDataQ') && config.ekfUseDataQ
                if isfield(data, 'Q')
                    qMat = squeeze(data.Q(:, :, noiseIdx));
                    if isequal(size(qMat), [2, 2])
                        qMat = 0.5 * (qMat + qMat');
                        obj.Q = qScale * qMat;
                    end
                end
            end

            if isfield(config, 'ekfUseDataP0') && config.ekfUseDataP0
                if isfield(data, 'P0')
                    p0 = squeeze(data.P0(:, :, noiseIdx));
                    if isequal(size(p0), [2, 2])
                        p0 = 0.5 * (p0 + p0');
                        obj.initCovariance = p0;
                    end
                end
            end

            if isfield(config, 'ekfInitCovariance')
                if isequal(size(config.ekfInitCovariance), [2, 2])
                    obj.initCovariance = config.ekfInitCovariance;
                end
            end

            if isfield(config, 'ekfEnabled')
                obj.ekfEnabled = config.ekfEnabled;
            end

            obj.particleCovariances = repmat(obj.initCovariance, [1, 1, obj.numParticles]);
        end

        function [state, est] = step(obj, state, iterIdx, pointIdx)
            if pointIdx == 3
                obj.particleCovariances = repmat(obj.initCovariance, [1, 1, obj.numParticles]);
            end

            xPrev = state.particlesPrev;
            vPrev = state.velPrev;
            muTrans = xPrev + vPrev + obj.processBias;
            zNow = obj.z(:, pointIdx, iterIdx);

            if obj.ekfEnabled
                [particlesProp, proposalCov, logProposal] = obj.sampleEkfProposal(muTrans, obj.particleCovariances, zNow);
                logLike = obj.computeLogLikelihood(zNow, obj.H_nonlinear(particlesProp));
                logPrior = obj.computeLogTransitionPrior(particlesProp, muTrans);
                logW = log(max(state.weights(:), 1e-300)) + logLike(:) + logPrior(:) - logProposal(:);
            else
                particlesProp = muTrans + obj.sampleProcess();
                proposalCov = obj.particleCovariances;
                logLike = obj.computeLogLikelihood(zNow, obj.H_nonlinear(particlesProp));
                logW = log(max(state.weights(:), 1e-300)) + logLike(:);
            end

            logW = logW - max(logW);
            weightsUpd = exp(logW);
            weightsUpd = weightsUpd + 1e-300;
            weightsUpd = weightsUpd / sum(weightsUpd);

            est = particlesProp * weightsUpd;

            [particlesRes, weightsRes, idxResampled, didResample] = obj.resampleEssWithIndices(particlesProp, weightsUpd);
            if didResample
                obj.particleCovariances = proposalCov(:, :, idxResampled);
            else
                obj.particleCovariances = proposalCov;
            end

            if didResample
                state.velPrev = particlesRes - state.particlesPrev(:, idxResampled);
            else
                state.velPrev = particlesRes - state.particlesPrev;
            end
            state.particlesPrev = particlesRes;
            state.weights = weightsRes;
            state.estimatedPos(:, pointIdx) = est;
        end

        function [samples, covOut, logQ] = sampleEkfProposal(obj, muTrans, covPrev, zNow)
            N = size(muTrans, 2);
            numAnchors = size(zNow, 1);
            Rk = (obj.noiseScale ^ 2) * eye(numAnchors);

            samples = zeros(2, N);
            covOut = zeros(2, 2, N);
            logQ = zeros(N, 1);

            for i = 1:N
                Ppred = covPrev(:, :, i) + obj.Q;
                Ppred = 0.5 * (Ppred + Ppred');

                xPred = muTrans(:, i);
                yPred = obj.H_nonlinear(xPred);
                Hk = obj.computeObservationJacobian(xPred);

                S = Hk * Ppred * Hk' + Rk;
                S = 0.5 * (S + S') + obj.regJitter * eye(numAnchors);

                K = (Ppred * Hk') / S;

                muQ = xPred + K * (zNow - yPred);

                I2 = eye(2);
                Pq = (I2 - K * Hk) * Ppred * (I2 - K * Hk)' + K * Rk * K';
                Pq = 0.5 * (Pq + Pq') + obj.regJitter * eye(2);

                xSample = obj.sampleGaussian(muQ, Pq);

                samples(:, i) = xSample;
                covOut(:, :, i) = Pq;
                logQ(i) = obj.logGaussianPdf(xSample, muQ, Pq);
            end
        end

        function Hk = computeObservationJacobian(obj, x)
            numAnchors = size(obj.anchorPos, 1);
            Hk = zeros(numAnchors, 2);

            for a = 1:numAnchors
                d = x - obj.anchorPos(a, :)';
                r = norm(d);
                if r < 1e-10
                    r = 1e-10;
                end
                Hk(a, :) = (d / r)';
            end
        end

        function logLike = computeLogLikelihood(obj, zNow, yPred)
            numAnchors = size(yPred, 1);
            invVar = 1 / (obj.noiseScale ^ 2);
            err = zNow - yPred;
            maha = sum((err .^ 2), 1) * invVar;
            const = numAnchors * log(2 * pi * (obj.noiseScale ^ 2));
            logLike = -0.5 * (maha + const);
        end

        function logPrior = computeLogTransitionPrior(obj, particles, muTrans)
            N = size(particles, 2);
            logPrior = zeros(N, 1);
            for i = 1:N
                logPrior(i) = obj.logGaussianPdf(particles(:, i), muTrans(:, i), obj.Q);
            end
        end

        function x = sampleGaussian(obj, mu, Sigma)
            L = obj.robustCholesky(Sigma);
            x = mu + L * randn(size(mu, 1), 1);
        end

        function logp = logGaussianPdf(obj, x, mu, Sigma)
            n = size(mu, 1);
            L = obj.robustCholesky(Sigma);
            diff = x - mu;
            y = L \ diff;
            maha = y' * y;
            logDet = 2 * sum(log(abs(diag(L))));
            logp = -0.5 * (maha + logDet + n * log(2 * pi));
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
