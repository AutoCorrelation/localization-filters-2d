classdef RougheningPriorEditingParticleFilter < NonlinearParticleFilter
    % RougheningPriorEditingParticleFilter
    %
    % Nonlinear PF with two heuristic compensation mechanisms:
    % 1) Prior editing (6-sigma gate with reject-and-repropagate loop)
    % 2) Roughening after resampling to reduce particle collapse

    properties
        rougheningK      (1,1) double = 0.2
        priorSigmaGate   (1,1) double = 6.0
        priorMaxRetry    (1,1) double = 30
    end

    methods
        function obj = RougheningPriorEditingParticleFilter(data, config, noiseIdx)
            obj@NonlinearParticleFilter(data, config, noiseIdx);

            if isfield(config, 'rougheningK')
                obj.rougheningK = max(config.rougheningK, 0);
            end
            if isfield(config, 'priorSigmaGate')
                obj.priorSigmaGate = max(config.priorSigmaGate, 0);
            end
            if isfield(config, 'priorMaxRetry')
                obj.priorMaxRetry = max(1, round(config.priorMaxRetry));
            end
        end

        function [state, est] = step(obj, state, iterIdx, pointIdx)
            particlesPred = state.particlesPrev + state.velPrev + obj.processBias + obj.sampleProcess();

            zNow = obj.z(:, pointIdx, iterIdx);
            particlesPred = obj.applyPriorEditing(particlesPred, state.particlesPrev, state.velPrev, zNow);

            weightsUpd = obj.updateWeightsNonlinear(particlesPred, state.weights, zNow);
            est = particlesPred * weightsUpd;

            [particlesRes, weightsRes, idxResampled, didResample] = obj.resampleEssWithIndices(particlesPred, weightsUpd);
            particlesRes = obj.applyRoughening(particlesRes);

            if didResample
                state.velPrev = particlesRes - state.particlesPrev(:, idxResampled);
            else
                state.velPrev = particlesRes - state.particlesPrev;
            end
            state.particlesPrev = particlesRes;
            state.weights = weightsRes;
            state.estimatedPos(:, pointIdx) = est;
        end

        function particlesOut = applyPriorEditing(obj, particlesIn, prevParticles, velPrev, zNow)
            particlesOut = particlesIn;
            gate = obj.priorSigmaGate * obj.noiseStd;

            validMask = obj.isWithinGate(particlesOut, zNow, gate);
            rejectIdx = find(~validMask);

            if isempty(rejectIdx)
                return;
            end

            sigmaRough = obj.computeRougheningSigma(particlesIn);

            for ii = 1:numel(rejectIdx)
                idx = rejectIdx(ii);
                velOne = velPrev(:, idx);
                prevOne = prevParticles(:, idx);

                bestParticle = particlesOut(:, idx);
                bestScore = obj.particleResidualScore(bestParticle, zNow);

                accepted = false;
                attempt = 0;
                while (~accepted) && (attempt < obj.priorMaxRetry)
                    attempt = attempt + 1;

                    % Repropagate from previous state particle with roughening.
                    base = prevOne + sigmaRough .* randn(2, 1);
                    candidate = base + velOne + obj.processBias + obj.sampleProcessSingle();

                    score = obj.particleResidualScore(candidate, zNow);
                    if score < bestScore
                        bestScore = score;
                        bestParticle = candidate;
                    end

                    accepted = obj.isWithinGate(candidate, zNow, gate);
                    if accepted
                        particlesOut(:, idx) = candidate;
                    end
                end

                if ~accepted
                    particlesOut(:, idx) = bestParticle;
                end
            end
        end

        function particlesOut = applyRoughening(obj, particlesIn)
            sigma = obj.computeRougheningSigma(particlesIn);
            particlesOut = particlesIn + sigma .* randn(size(particlesIn));
        end

        function sigma = computeRougheningSigma(obj, particles)
            n = size(particles, 1);
            N = size(particles, 2);

            maxDiff = max(particles, [], 2) - min(particles, [], 2);
            sigma = obj.rougheningK * maxDiff * (N ^ (-1 / n));
        end

        function noise = sampleProcessSingle(obj)
            if isempty(obj.processNoise)
                noise = obj.noiseStd * randn(2, 1);
                return;
            end

            idx = randi(size(obj.processNoise, 2));
            noise = obj.processNoise(:, idx);
        end

        function score = particleResidualScore(obj, particle, zNow)
            yPred = obj.H_nonlinear(particle);
            residual = zNow - yPred;
            score = max(abs(residual));
        end

        function tf = isWithinGate(obj, particles, zNow, gate)
            yPred = obj.H_nonlinear(particles);
            residual = zNow - yPred;
            tf = all(abs(residual) <= gate, 1);
        end
    end
end
