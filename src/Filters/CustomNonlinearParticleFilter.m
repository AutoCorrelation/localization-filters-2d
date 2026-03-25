classdef CustomNonlinearParticleFilter < NonlinearParticleFilter
    properties
        % Custom properties for the new filter can be added here
    end

    methods
        function obj = CustomNonlinearParticleFilter(data, config, noiseIdx)
            obj@NonlinearParticleFilter(data, config, noiseIdx);
            % Additional custom parameters/logic can be implemented here
        end

        function state = initializeState(obj, numPoints)
            state = initializeState@NonlinearParticleFilter(obj, numPoints);
            state.M = zeros(4, obj.numParticles); % AdaBelief 1st moment
            state.S = zeros(4, obj.numParticles); % AdaBelief 2nd moment
            % Add custom initialization logic if needed
        end

        function weights = updateWeightsNonlinear(obj, particles, prevWeights, zNow)
            yPred = obj.H_nonlinear(particles);
            numAnchors = size(yPred, 1);
            Rinv = eye(numAnchors) / (obj.noiseStd^2) / 1;
            errors = zNow - yPred;
            distances = sum((Rinv * errors) .* errors, 1);

            weights = prevWeights(:)' .* exp(-0.5 * distances);
            weights = weights + 1e-300;
            weights = (weights / sum(weights)).';
        end

        % function [state, est] = step(obj, state, iterIdx, pointIdx)
        %     particlesPred = state.particlesPrev + state.velPrev + obj.sampleProcess();

        %     zNow = obj.z(:, pointIdx, iterIdx);
        %     weightsUpd = obj.updateWeightsNonlinear(particlesPred, state.weights, zNow);

        %     est = particlesPred * weightsUpd;
        %     [particlesRes, weightsRes] = obj.resampleEss(particlesPred, weightsUpd);

        %     state.velPrev = particlesRes - state.particlesPrev;
        %     state.particlesPrev = particlesRes;
        %     state.weights = weightsRes;
        %     state.estimatedPos(:, pointIdx) = est;
        % end

        function [outM, outS] = belief(obj, z, xparticles, inM, inS)
            % Global AdaBelief moments calculation for weighted estimate
            % xparticles: 2 x 1 (weighted particle estimate)
            % z: 4 x 1 (measurement vector)
            % H: (nonlinear observation function)
            % m, s: 4 x 1 (1st and 2nd moments of innovation)

            beta1 = 0.9;
            beta2 = 0.999;

            % Innovation: per-particle residual norm (1 x N)
            residual = z - obj.H_nonlinear(xparticles); % 4 x N
            g = residual
            % Update moments with exponential moving average
            inM = beta1 * inM + (1 - beta1) * g;
            inS = beta2 * inS + (1 - beta2) * ((g - inM).^2) + 1e-8;

            % Bias correction
            outM = inM / (1 - beta1);
            outS = inS / (1 - beta2);
        end
    end
end
