classdef ResidualSquaredAdaptiveParticleFilter < AdaptiveParticleFilter
    % ResidualSquaredAdaptiveParticleFilter
    % e^2 기반(?�차 ?�곱 EMA) R-inflation PF.
    %
    % AdaptiveParticleFilter(AdaBelief???�??비교�??�한 분리 구현:
    %   s_k(i) = beta * s_{k-1}(i) + (1-beta) * e_k(i)^2

    methods
        function obj = ResidualSquaredAdaptiveParticleFilter(data, config, noiseIdx, beta, lambdaR)
            obj@AdaptiveParticleFilter(data, config, noiseIdx, beta, lambdaR);
        end

        function [state, est] = step(obj, state, iterIdx, pointIdx)
            particlesPred = state.particlesPrev + state.velPrev + obj.processBias + obj.sampleProcess();

            zNow = obj.z(:, pointIdx, iterIdx);

            xHatWeighted = particlesPred * state.weights;
            yPredWeighted = obj.H_nonlinear(xHatWeighted);
            e = zNow - yPredWeighted;

            % Residual energy EMA (e^2 form).
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
        end
    end
end
