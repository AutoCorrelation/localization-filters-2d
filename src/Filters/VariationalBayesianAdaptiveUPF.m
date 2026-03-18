classdef VariationalBayesianAdaptiveUPF < NonlinearParticleFilter
    % VariationalBayesianAdaptiveUPF
    % VB-based adaptive Unscented PF with Inverse-Gamma noise adaptation.

    properties
        vbIterations            (1,1) double = 3
        alpha0                  (1,1) double = 2.0
        beta0                   (1,1) double = 1.0
        vbForgettingFactor      (1,1) double = 0.99
        vbTolerance             (1,1) double = 1e-3
        rFloor                  (1,1) double = 1e-6
        rCeil                   (1,1) double = 1e3

        proposalPriorScale      (1,1) double = 1.0
        covJitter               (1,1) double = 1e-8

        utAlpha                 (1,1) double = 1e-1
        utBeta                  (1,1) double = 2.0
        utKappa                 (1,1) double = 0.0
    end

    properties (Access = private)
        utWm                    (1,:) double = zeros(1, 5)
        utWc                    (1,:) double = zeros(1, 5)
        utScale                 (1,1) double = 1.0
    end

    methods
        function obj = VariationalBayesianAdaptiveUPF(data, config, noiseIdx)
            obj@NonlinearParticleFilter(data, config, noiseIdx);

            if isfield(config, 'vbUpfIterations')
                obj.vbIterations = max(1, round(config.vbUpfIterations));
            end
            if isfield(config, 'vbUpfAlpha0')
                obj.alpha0 = config.vbUpfAlpha0;
            end
            if isfield(config, 'vbUpfBeta0')
                obj.beta0 = config.vbUpfBeta0;
            end
            if isfield(config, 'vbUpfRFloor')
                obj.rFloor = config.vbUpfRFloor;
            end
            if isfield(config, 'vbUpfRCeil')
                obj.rCeil = config.vbUpfRCeil;
            end
            if isfield(config, 'vbUpfForgettingFactor')
                obj.vbForgettingFactor = min(max(config.vbUpfForgettingFactor, 0.0), 1.0);
            end
            if isfield(config, 'vbUpfTolerance')
                obj.vbTolerance = max(config.vbUpfTolerance, 0.0);
            end
            if isfield(config, 'vbUpfProposalPriorScale')
                obj.proposalPriorScale = config.vbUpfProposalPriorScale;
            end
            if isfield(config, 'vbUpfCovJitter')
                obj.covJitter = config.vbUpfCovJitter;
            end
            if isfield(config, 'vbUpfUtAlpha')
                obj.utAlpha = config.vbUpfUtAlpha;
            end
            if isfield(config, 'vbUpfUtBeta')
                obj.utBeta = config.vbUpfUtBeta;
            end
            if isfield(config, 'vbUpfUtKappa')
                obj.utKappa = config.vbUpfUtKappa;
            end

            [obj.utWm, obj.utWc, obj.utScale] = obj.computeUtWeights();
        end

        function state = initializeState(obj, numPoints)
            state = initializeState@NonlinearParticleFilter(obj, numPoints);
            numAnchors = size(obj.anchorPos, 1);
            state.vbAlpha = obj.alpha0 * ones(numAnchors, obj.numParticles);
            state.vbBeta = obj.beta0 * ones(numAnchors, obj.numParticles);
        end

        function [state, est] = step(obj, state, iterIdx, pointIdx)
            particlesPred = state.particlesPrev + state.velPrev + obj.processBias + obj.sampleProcess();
            zNow = obj.z(:, pointIdx, iterIdx);

            [proposalMean, proposalCov, rDiagParticles, alphaParticles, betaParticles] = ...
                obj.vbUnscentedProposalBatch(particlesPred, zNow, state.vbAlpha, state.vbBeta);

            particlesProp = obj.sampleGaussianBatch(proposalMean, proposalCov);
            weightsUpd = obj.updateImportanceWeightsVB( ...
                particlesProp, particlesPred, state.weights, zNow, proposalMean, proposalCov, rDiagParticles);

            est = particlesProp * weightsUpd;
            [particlesRes, weightsRes, alphaRes, betaRes] = ...
                obj.resampleEssWithVBParams(particlesProp, weightsUpd, alphaParticles, betaParticles);

            state.velPrev = est * ones(1, obj.numParticles) - state.particlesPrev;
            state.particlesPrev = particlesRes;
            state.weights = weightsRes;
            state.vbAlpha = alphaRes;
            state.vbBeta = betaRes;
            state.estimatedPos(:, pointIdx) = est;
        end

        function [proposalMean, proposalCov, rDiagParticles, alphaParticles, betaParticles] = ...
                vbUnscentedProposalBatch(obj, particlesPred, zNow, alphaPrev, betaPrev)
            N = size(particlesPred, 2);
            nz = numel(zNow);

            proposalMean = zeros(2, N);
            proposalCov = zeros(2, 2, N);
            rDiagParticles = zeros(nz, N);
            alphaParticles = alphaPrev;
            betaParticles = betaPrev;

            priorCov = (obj.proposalPriorScale * obj.noiseScale ^ 2) * eye(2);
            priorCov = priorCov + obj.covJitter * eye(2);

            for i = 1:N
                mu = particlesPred(:, i);
                P = priorCov;

                alphaVec = alphaPrev(:, i);
                betaVec = betaPrev(:, i);
                rDiag = min(max(betaVec ./ max(alphaVec, obj.covJitter), obj.rFloor), obj.rCeil);

                for it = 1:obj.vbIterations
                    sigmaPts = obj.generateSigmaPointsFast(mu, P);
                    ySigma = obj.H_nonlinear(sigmaPts);

                    yMean = ySigma * obj.utWm.';
                    yDiff = ySigma - yMean;
                    xDiff = sigmaPts - mu;

                    Pyy = (yDiff .* obj.utWc) * yDiff.';
                    Pxy = (xDiff .* obj.utWc) * yDiff.';
                    Pyy = 0.5 * (Pyy + Pyy.') + obj.covJitter * eye(size(Pyy, 1));

                    innov = zNow - yMean;
                    SexpDiag = diag(Pyy + innov * innov.');

                    alphaNew = obj.vbForgettingFactor * alphaVec + (1 - obj.vbForgettingFactor) * obj.alpha0 + 0.5;
                    betaNew = obj.vbForgettingFactor * betaVec + (1 - obj.vbForgettingFactor) * obj.beta0 + 0.5 * SexpDiag;
                    rNew = min(max(betaNew ./ max(alphaNew, obj.covJitter), obj.rFloor), obj.rCeil);

                    relChange = max(abs(rNew - rDiag) ./ max(rDiag, obj.rFloor));
                    alphaVec = alphaNew;
                    betaVec = betaNew;
                    rDiag = rNew;

                    S = Pyy + diag(rDiag) + obj.covJitter * eye(size(Pyy, 1));
                    K = Pxy / S;

                    mu = mu + K * innov;
                    P = P - K * S * K.';
                    P = obj.projectCov2x2(P);

                    if relChange < obj.vbTolerance
                        break;
                    end
                end

                proposalMean(:, i) = mu;
                proposalCov(:, :, i) = P;
                rDiagParticles(:, i) = rDiag;
                alphaParticles(:, i) = alphaVec;
                betaParticles(:, i) = betaVec;
            end
        end

        function particles = sampleGaussianBatch(obj, meanBatch, covBatch)
            N = size(meanBatch, 2);
            particles = zeros(2, N);

            for i = 1:N
                Pi = obj.projectCov2x2(covBatch(:, :, i));
                Li = obj.safeChol2x2(Pi);
                particles(:, i) = meanBatch(:, i) + Li * randn(2, 1);
            end
        end

        function weights = updateImportanceWeightsVB(obj, particles, particlesPred, prevWeights, zNow, proposalMean, proposalCov, rDiagParticles)
            yPred = obj.H_nonlinear(particles);
            residual = zNow - yPred;

            logLike = obj.gaussianLogPdfDiagBatch(residual, rDiagParticles);

            priorCov = (obj.proposalPriorScale * obj.noiseScale ^ 2) * eye(2);
            priorCov = priorCov + obj.covJitter * eye(2);
            logPrior = obj.gaussianLogPdf2DConstantCov(particles, particlesPred, priorCov);
            logProposal = obj.gaussianLogPdf2DBatch(particles, proposalMean, proposalCov);

            logWeights = log(max(prevWeights(:), 1e-300)) + logLike + logPrior - logProposal;
            logWeights = logWeights - max(logWeights);

            weights = exp(logWeights);
            weights = weights + 1e-300;
            weights = weights / sum(weights);
        end

        function logp = gaussianLogPdfDiagBatch(obj, residual, diagVar)
            safeVar = min(max(diagVar, obj.rFloor), obj.rCeil);
            quadTerm = sum((residual .^ 2) ./ safeVar, 1).';
            logDet = sum(log(safeVar), 1).';
            dim = size(residual, 1);
            logp = -0.5 * (dim * log(2 * pi) + logDet + quadTerm);
        end

        function logp = gaussianLogPdf2DConstantCov(~, x, mu, cov2x2)
            dx = x(1, :) - mu(1, :);
            dy = x(2, :) - mu(2, :);

            a = cov2x2(1, 1);
            b = cov2x2(1, 2);
            c = cov2x2(2, 2);

            detCov = max(a * c - b * b, 1e-12);
            quad = (c * (dx .^ 2) - 2 * b * (dx .* dy) + a * (dy .^ 2)) / detCov;
            logp = -0.5 * (2 * log(2 * pi) + log(detCov) + quad.');
        end

        function logp = gaussianLogPdf2DBatch(~, x, mu, covBatch)
            dx = x(1, :) - mu(1, :);
            dy = x(2, :) - mu(2, :);

            a = squeeze(covBatch(1, 1, :));
            b = squeeze(covBatch(1, 2, :));
            c = squeeze(covBatch(2, 2, :));

            detCov = a .* c - b .* b;
            detCov = max(detCov, 1e-12);

            quad = (c .* (dx .^ 2).' - 2 * b .* (dx .* dy).' + a .* (dy .^ 2).') ./ detCov;
            logp = -0.5 * (2 * log(2 * pi) + log(detCov) + quad);
        end

        function [sigmaPts, Wm, Wc] = generateSigmaPoints(obj, mu, P)
            L = numel(mu);
            lambda = obj.utAlpha ^ 2 * (L + obj.utKappa) - L;
            scale = L + lambda;
            if scale <= 1e-8
                scale = 1e-8;
            end

            cholP = obj.safeChol2x2(P);
            spread = sqrt(scale) * cholP;

            sigmaPts = [mu, mu + spread(:, 1), mu + spread(:, 2), mu - spread(:, 1), mu - spread(:, 2)];

            Wm = [lambda / scale, repmat(1 / (2 * scale), 1, 2 * L)];
            Wc = Wm;
            Wc(1) = Wc(1) + (1 - obj.utAlpha ^ 2 + obj.utBeta);
        end

        function sigmaPts = generateSigmaPointsFast(obj, mu, P)
            cholP = obj.safeChol2x2(P);
            spread = sqrt(obj.utScale) * cholP;
            sigmaPts = [mu, mu + spread(:, 1), mu + spread(:, 2), mu - spread(:, 1), mu - spread(:, 2)];
        end

        function P = projectCov2x2(obj, Pin)
            P = 0.5 * (Pin + Pin.');
            [V, D] = eig(P);
            eigVals = diag(D);
            eigVals = max(eigVals, obj.covJitter);
            P = V * diag(eigVals) * V.';
            P = 0.5 * (P + P.') + obj.covJitter * eye(2);
        end

        function L = safeChol2x2(obj, P)
            P = obj.projectCov2x2(P);

            [L, cholFlag] = chol(P, 'lower');
            if cholFlag == 0
                return;
            end

            jitter = obj.covJitter;
            for k = 1:5
                [L, cholFlag] = chol(P + jitter * eye(2), 'lower');
                if cholFlag == 0
                    return;
                end
                jitter = jitter * 10;
            end

            L = chol(P + jitter * eye(2), 'lower');
        end

        function [weightsMean, weightsCov, scale] = computeUtWeights(obj)
            L = 2;
            lambda = obj.utAlpha ^ 2 * (L + obj.utKappa) - L;
            scale = L + lambda;
            if scale <= 1e-8
                scale = 1e-8;
            end

            weightsMean = [lambda / scale, repmat(1 / (2 * scale), 1, 2 * L)];
            weightsCov = weightsMean;
            weightsCov(1) = weightsCov(1) + (1 - obj.utAlpha ^ 2 + obj.utBeta);
        end

        function [particlesOut, weightsOut, alphaOut, betaOut] = ...
                resampleEssWithVBParams(obj, particles, weights, alphaIn, betaIn)
            N = obj.numParticles;
            ess = 1 / sum(weights .^ 2);

            if ess >= N * obj.resampleThresholdRatio
                particlesOut = particles;
                weightsOut = weights;
                alphaOut = alphaIn;
                betaOut = betaIn;
                return;
            end

            cdf = cumsum(weights(:));
            u0 = rand / N;
            u = u0 + (0:N-1)' / N;
            idx = zeros(N, 1);
            j = 1;
            for i = 1:N
                while (j < N) && (u(i) > cdf(j))
                    j = j + 1;
                end
                idx(i) = j;
            end

            particlesOut = particles(:, idx);
            alphaOut = alphaIn(:, idx);
            betaOut = betaIn(:, idx);
            weightsOut = ones(N, 1) / N;
        end
    end
end