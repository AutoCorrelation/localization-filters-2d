classdef RDiagPriorEditAdaptiveParticleFilter < AdaptiveParticleFilter
    % RDiagPriorEditAdaptiveParticleFilter
    % R inflation으로 얻은 diag(R_k)를 prior editing gate에 사용한다.

    properties
        priorSigmaGate  (1,1) double = 6.0
        priorMaxRetry   (1,1) double = 20
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
        end

        function [state, est] = step(obj, state, iterIdx, pointIdx)
            particlesPred = state.particlesPrev + state.velPrev + obj.processBias + obj.sampleProcess();

            zNow = obj.z(:, pointIdx, iterIdx);
            xHatWeighted = particlesPred * state.weights;
            yPredWeighted = obj.H_nonlinear(xHatWeighted);
            e = zNow - yPredWeighted;

            state.sMoment = obj.beta * state.sMoment + (1 - obj.beta) * (e .^ 2);
            state.diagR = state.nominalDiagR + obj.lambdaR * state.sMoment;
            state.diagR = min(max(state.diagR, obj.rFloor), obj.rCeil);

            particlesEdited = obj.applyPriorEditingWithAdaptiveR(particlesPred, state.velPrev, zNow, state.diagR);

            weightsUpd = obj.updateWeightsWithR(particlesEdited, state.weights, zNow, diag(state.diagR));

            est = particlesEdited * weightsUpd;
            [particlesRes, weightsRes] = obj.resampleEss(particlesEdited, weightsUpd);

            state.velPrev = est * ones(1, obj.numParticles) - state.particlesPrev;
            state.particlesPrev = particlesRes;
            state.weights = weightsRes;
            state.estimatedPos(:, pointIdx) = est;
        end

        function particlesOut = applyPriorEditingWithAdaptiveR(obj, particlesIn, velPrev, zNow, diagR)
            particlesOut = particlesIn;

            sigmaGate = obj.priorSigmaGate * sqrt(max(diagR(:), obj.rFloor));
            validMask = obj.isWithinAdaptiveGate(particlesOut, zNow, sigmaGate);
            rejectIdx = find(~validMask);

            for ii = 1:numel(rejectIdx)
                idx = rejectIdx(ii);
                bestParticle = particlesOut(:, idx);
                bestScore = obj.normalizedResidualScore(bestParticle, zNow, sigmaGate);

                accepted = false;
                for attempt = 1:obj.priorMaxRetry
                    candidate = particlesOut(:, idx) + velPrev(:, idx) + obj.processBias + obj.sampleProcessSingle();
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
                noise = obj.noiseScale * randn(2, 1);
                return;
            end

            idx = randi(size(obj.processNoise, 2));
            noise = obj.processNoise(:, idx);
        end
    end
end