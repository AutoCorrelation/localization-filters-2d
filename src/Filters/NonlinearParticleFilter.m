classdef NonlinearParticleFilter < LinearParticleFilter
    properties
        anchorPos
        Rpoint
    end

    methods
        function obj = NonlinearParticleFilter(data, config, noiseIdx)
            obj@LinearParticleFilter(data, config, noiseIdx);
            obj.z = squeeze(data.ranging(:, :, :, noiseIdx));
            obj.R = 4* config.noiseVariance(noiseIdx) * eye(size(data.ranging, 1));
            obj.anchorPos = config.Anchor';
            obj.Rpoint = [];
            if isfield(data, 'R_corrected_point') && ~isempty(data.R_corrected_point)
                rcp = data.R_corrected_point;
                if ndims(rcp) == 4
                    obj.Rpoint = squeeze(rcp(:, :, :, noiseIdx));
                elseif ndims(rcp) == 3
                    obj.Rpoint = rcp;
                end
            end
        end

        function [state, est] = step(obj, state, iterIdx, pointIdx)
            particlesPred = state.particlesPrev + state.velPrev + obj.processBias + obj.sampleProcess();

            zNow = obj.z(:, pointIdx, iterIdx);
            weightsUpd = obj.updateWeightsNonlinear(particlesPred, state.weights, zNow, pointIdx);

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

        function weights = updateWeightsNonlinear(obj, particles, prevWeights, zNow, pointIdx)
            yPred = obj.H_nonlinear(particles);
            numAnchors = size(yPred, 1);
            errors = zNow - yPred;

            if nargin >= 5 && ~isempty(obj.Rpoint) && pointIdx <= size(obj.Rpoint, 3)
                Rk = obj.Rpoint(:, :, pointIdx);
                % Corrected scenario: R is treated as diagonal.
                % Use reciprocal of diagonal entries (with floor for stability).
                rdiag = diag(Rk);
                rdiag = max(real(rdiag), 1e-8);
                rinvDiag = 1 ./ rdiag;
                distances = sum((errors .^ 2) .* rinvDiag, 1);
            else
                Rinv = eye(numAnchors) / (obj.noiseStd^2);
                distances = sum((Rinv * errors) .* errors, 1);
            end

            % Log-domain update to avoid under/overflow.
            logPrev = log(prevWeights(:)' + 1e-300);
            logW = logPrev - 0.5 * distances;
            logW = logW - max(logW);
            weights = exp(logW);
            weights = weights / sum(weights);
            weights = weights(:);
        end

    end
end