classdef BeliefQShrinkAdaptiveParticleFilter < AdaptiveParticleFilter
    % BeliefQShrinkAdaptiveParticleFilter
    % s_k 기반?�로 R inflation???��??�면???�측 ?�이�?Q)�?축소?�다.

    properties
        qShrinkGain      (1,1) double = 0.3
        qShrinkMinScale  (1,1) double = 0.35
    end

    methods
        function obj = BeliefQShrinkAdaptiveParticleFilter(data, config, noiseIdx, beta, lambdaR)
            obj@AdaptiveParticleFilter(data, config, noiseIdx, beta, lambdaR);

            if isfield(config, 'beliefQShrinkGain')
                obj.qShrinkGain = max(config.beliefQShrinkGain, 0);
            end
            if isfield(config, 'beliefQShrinkMinScale')
                obj.qShrinkMinScale = max(config.beliefQShrinkMinScale, 1e-3);
            end
        end

        function state = initializeState(obj, numPoints)
            state = initializeState@AdaptiveParticleFilter(obj, numPoints);
            state.qScaleNow = 1.0;
        end

        function [state, est] = step(obj, state, iterIdx, pointIdx)
            beliefRatio = mean(state.sMoment ./ max(state.nominalDiagR, obj.rFloor));
            qVarScale = 1 / (1 + obj.qShrinkGain * beliefRatio);
            qVarScale = max(qVarScale, obj.qShrinkMinScale ^ 2);
            qScale = sqrt(qVarScale);

            particlesPred = state.particlesPrev + state.velPrev + obj.processBias + qScale * obj.sampleProcess();

            zNow = obj.z(:, pointIdx, iterIdx);
            xHatWeighted = particlesPred * state.weights;
            yPredWeighted = obj.H_nonlinear(xHatWeighted);
            e = zNow - yPredWeighted;

            state.sMoment = obj.beta * state.sMoment + (1 - obj.beta) * (e .^ 2);
            state.diagR = state.nominalDiagR + obj.lambdaR * state.sMoment;
            state.diagR = min(max(state.diagR, obj.rFloor), obj.rCeil);

            weightsUpd = obj.updateWeightsWithR(particlesPred, state.weights, zNow, diag(state.diagR));

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
            state.qScaleNow = qScale;
        end
    end
end