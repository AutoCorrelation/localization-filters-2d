classdef RDiagPriorEditAdaptiveParticleFilter < AdaptiveParticleFilter

    properties
        priorSigmaGate  (1,1) double = 6.0
        priorMaxRetry   (1,1) double = 20
        rougheningK     (1,1) double = 0.2
    end

    methods
        function obj = RDiagPriorEditAdaptiveParticleFilter(data, config, noiseIdx, beta, lambdaR)
            obj@AdaptiveParticleFilter(data, config, noiseIdx, beta, lambdaR);

            if isfield(config, 'rdiagPriorSigmaGate')
                obj.priorSigmaGate = max(config.rdiagPriorSigmaGate, 0);
            end
            if isfield(config, 'rdiagPriorMaxRetry')
                obj.priorMaxRetry = max(1, round(config.rdiagPriorMaxRetry));
            end
            if isfield(config, 'rdiagRougheningK')
                obj.rougheningK = max(config.rdiagRougheningK, 0);
            end
        end

        function [state, est] = step(obj, state, iterIdx, pointIdx)
            particlesPred = state.particlesPrev + state.velPrev + obj.processBias + obj.sampleProcess();

            zNow = obj.z(:, pointIdx, iterIdx);
            xHatWeighted = particlesPred * state.weights;
            yPredWeighted = obj.H_nonlinear(xHatWeighted);
            e = zNow - yPredWeighted;

            state.mMoment = obj.beta * state.mMoment + (1 - obj.beta) * e;
            state.sMoment = obj.beta * state.sMoment + (1 - obj.beta) * ((e - state.mMoment) .^ 2);
            state.diagR = state.nominalDiagR + obj.lambdaR * state.sMoment;
            state.diagR = min(max(state.diagR, obj.rFloor), obj.rCeil);

            particlesEdited = obj.applyPriorEditingWithAdaptiveR( ...
                particlesPred, state.particlesPrev, state.velPrev, zNow, state.diagR);

            weightsUpd = obj.updateWeightsWithR(particlesEdited, state.weights, zNow, diag(state.diagR));

            est = particlesEdited * weightsUpd;
            [particlesRes, weightsRes, idxResampled, didResample] = obj.resampleEssWithIndices(particlesEdited, weightsUpd);

            if didResample
                state.velPrev = particlesRes - state.particlesPrev(:, idxResampled);
            else
                state.velPrev = particlesRes - state.particlesPrev;
            end
            state.particlesPrev = particlesRes;
            state.weights = weightsRes;
            state.estimatedPos(:, pointIdx) = est;
        end

        function particlesOut = applyPriorEditingWithAdaptiveR(obj, particlesIn, prevParticles, velPrev, zNow, diagR)
            particlesOut = particlesIn;

            sigmaGate = obj.priorSigmaGate * sqrt(max(diagR(:), obj.rFloor));
            validMask = obj.isWithinAdaptiveGate(particlesOut, zNow, sigmaGate);
            rejectIdx = find(~validMask);
            sigmaRough = obj.computeRougheningSigma(particlesIn);

            for ii = 1:numel(rejectIdx)
                idx = rejectIdx(ii);
                bestParticle = particlesOut(:, idx);
                bestScore = obj.normalizedResidualScore(bestParticle, zNow, sigmaGate);

                accepted = false;
                for attempt = 1:obj.priorMaxRetry
                    base = prevParticles(:, idx) + sigmaRough .* randn(2, 1);
                    candidate = base + velPrev(:, idx) + obj.processBias + obj.sampleProcessSingle();
                    score = obj.normalizedResidualScore(candidate, zNow, sigmaGate);

                    if score < bestScore
                        bestScore = score;
                        bestParticle = candidate;
                    end

                    if obj.isWithinAdaptiveGate(candidate, zNow, sigmaGate)
                        particlesOut(:, idx) = candidate;
                        accepted = true;
                        break;
                    end
                end

                if ~accepted
                    particlesOut(:, idx) = bestParticle;
                end
            end
        end

        function sigma = computeRougheningSigma(obj, particles)
            n = size(particles, 1);
            N = size(particles, 2);
            span = max(particles, [], 2) - min(particles, [], 2);
            sigma = obj.rougheningK * span * (N ^ (-1 / n));
        end

        function tf = isWithinAdaptiveGate(obj, particles, zNow, sigmaGate)
            yPred = obj.H_nonlinear(particles);
            residual = abs(zNow - yPred);
            tf = all(residual <= sigmaGate, 1);
        end

        function score = normalizedResidualScore(obj, particle, zNow, sigmaGate)
            yPred = obj.H_nonlinear(particle);
            residual = abs(zNow - yPred);
            score = sum(residual ./ max(sigmaGate, obj.rFloor));
        end

        function noise = sampleProcessSingle(obj)
            if isempty(obj.processNoise)
                noise = obj.noiseStd * randn(2, 1);
                return;
            end

            idx = randi(size(obj.processNoise, 2));
            noise = obj.processNoise(:, idx);
        end
    end
end