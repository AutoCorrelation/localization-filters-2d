classdef BeliefRougheningAdaptiveParticleFilter < AdaptiveParticleFilter
    % BeliefRougheningAdaptiveParticleFilter
    % s_k에 따라 roughening 강도를 동적으로 조절한다.

    properties
        rougheningKBase  (1,1) double = 0.2
        rougheningGain   (1,1) double = 0.6
        rougheningKMax   (1,1) double = 1.5
    end

    methods
        function obj = BeliefRougheningAdaptiveParticleFilter(data, config, noiseIdx, beta, lambdaR)
            obj@AdaptiveParticleFilter(data, config, noiseIdx, beta, lambdaR);

            if isfield(config, 'beliefRougheningKBase')
                obj.rougheningKBase = max(config.beliefRougheningKBase, 0);
            end
            if isfield(config, 'beliefRougheningGain')
                obj.rougheningGain = max(config.beliefRougheningGain, 0);
            end
            if isfield(config, 'beliefRougheningKMax')
                obj.rougheningKMax = max(config.beliefRougheningKMax, obj.rougheningKBase);
            end
        end

        function state = initializeState(obj, numPoints)
            state = initializeState@AdaptiveParticleFilter(obj, numPoints);
            state.rougheningKNow = obj.rougheningKBase;
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

            weightsUpd = obj.updateWeightsWithR(particlesPred, state.weights, zNow, diag(state.diagR));
            est = particlesPred * weightsUpd;

            ess = 1 / sum(weightsUpd .^ 2);
            didResample = ess < obj.numParticles * obj.resampleThresholdRatio;
            [particlesRes, weightsRes] = obj.resampleEss(particlesPred, weightsUpd);

            beliefRatio = mean(state.sMoment ./ max(state.nominalDiagR, obj.rFloor));
            rougheningKNow = obj.rougheningKBase * (1 + obj.rougheningGain * beliefRatio);
            rougheningKNow = min(rougheningKNow, obj.rougheningKMax);

            if didResample
                particlesRes = obj.applyBeliefRoughening(particlesRes, rougheningKNow);
            end

            state.velPrev = est * ones(1, obj.numParticles) - state.particlesPrev;
            state.particlesPrev = particlesRes;
            state.weights = weightsRes;
            state.estimatedPos(:, pointIdx) = est;
            state.rougheningKNow = rougheningKNow;
        end

        function particlesOut = applyBeliefRoughening(~, particlesIn, rougheningK)
            n = size(particlesIn, 1);
            N = size(particlesIn, 2);

            span = max(particlesIn, [], 2) - min(particlesIn, [], 2);
            sigma = rougheningK * span * (N ^ (-1 / n));
            particlesOut = particlesIn + sigma .* randn(size(particlesIn));
        end
    end
end