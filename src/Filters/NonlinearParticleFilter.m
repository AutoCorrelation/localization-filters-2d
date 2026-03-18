classdef NonlinearParticleFilter < LinearParticleFilter
    properties
        anchorPos
    end

    methods
        function obj = NonlinearParticleFilter(data, config, noiseIdx)
            obj@LinearParticleFilter(data, config, noiseIdx);
            obj.z = squeeze(data.ranging(:, :, :, noiseIdx));
            obj.R = config.noiseVariance(noiseIdx) * eye(size(data.ranging, 1));
            obj.anchorPos = config.Anchor';
        end

        function [state, est] = step(obj, state, iterIdx, pointIdx)
            particlesPred = state.particlesPrev + state.velPrev + obj.processBias + obj.sampleProcess();

            zNow = obj.z(:, pointIdx, iterIdx);
            weightsUpd = obj.updateWeightsNonlinear(particlesPred, state.weights, zNow);

            est = particlesPred * weightsUpd;
            [particlesRes, weightsRes] = obj.resampleEss(particlesPred, weightsUpd);

            state.velPrev = est * ones(1, obj.numParticles) - state.particlesPrev;
            state.particlesPrev = particlesRes;
            state.weights = weightsRes;
            state.estimatedPos(:, pointIdx) = est;
        end

        function yPred = H_nonlinear(obj, x)
            % Nonlinear observation model: ranging from each anchor.
            numAnchors = size(obj.anchorPos, 1);
            numParticles = size(x, 2);
            yPred = zeros(numAnchors, numParticles);

            for i = 1:numAnchors
                dx = x(1, :) - obj.anchorPos(i, 1);
                dy = x(2, :) - obj.anchorPos(i, 2);
                yPred(i, :) = sqrt(dx.^2 + dy.^2);
            end
        end

        function weights = updateWeightsNonlinear(obj, particles, prevWeights, zNow)
            yPred = obj.H_nonlinear(particles);
            numAnchors = size(yPred, 1);
            Rinv = eye(numAnchors) / (obj.noiseScale^2);
            errors = zNow - yPred;
            distances = sum((Rinv * errors) .* errors, 1);

            weights = prevWeights(:)' .* exp(-0.5 * distances);
            weights = weights + 1e-300;
            weights = (weights / sum(weights)).';
        end

    end
end