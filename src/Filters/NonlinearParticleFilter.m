classdef NonlinearParticleFilter
    properties
        xHat
        z
        R
        processNoise
        toaNoise
        numParticles
        noiseScale
        anchorPos
    end

    methods
        function obj = NonlinearParticleFilter(data, config, noiseIdx)
            obj.xHat = squeeze(data.x_hat_LLS(:, :, :, noiseIdx));
            obj.z = squeeze(data.ranging(:, :, :, noiseIdx));
            obj.R = config.noiseVariance(noiseIdx) * eye(size(data.ranging, 1));

            obj.processNoise = localExtractNoiseBank(data.processNoise, noiseIdx);
            obj.toaNoise = localExtractNoiseBank(data.toaNoise, noiseIdx);

            if isfield(config, 'numParticles')
                obj.numParticles = config.numParticles;
            else
                obj.numParticles = 500;
            end

            obj.noiseScale = sqrt(config.noiseVariance(noiseIdx));
            
            if isfield(config, 'anchorPos')
                obj.anchorPos = config.anchorPos;
            else
                obj.anchorPos = [0, 10; 0, 0; 10, 0; 10, 10];
            end
        end

        function state = initializeState(obj, numPoints)
            state.estimatedPos = zeros(2, numPoints);
            state.particlesPrev = zeros(2, obj.numParticles);
            state.velPrev = zeros(2, obj.numParticles);
            state.weights = ones(obj.numParticles, 1) / obj.numParticles;
        end

        function [state, p1, p2] = initializeFirstTwo(obj, state, iterIdx)
            p1 = obj.xHat(:, 1, iterIdx);
            p2 = obj.xHat(:, 2, iterIdx);

            state.estimatedPos(:, 1) = p1;
            state.estimatedPos(:, 2) = p2;

            sampledPrev = obj.sampleToa(p1);
            sampledCurr = obj.sampleToa(p2);
            state.particlesPrev = sampledCurr;
            state.velPrev = sampledCurr - sampledPrev;
        end

        function [state, est] = step(obj, state, iterIdx, pointIdx)
            particlesPred = state.particlesPrev + state.velPrev + obj.sampleProcess();

            zNow = obj.z(:, pointIdx, iterIdx);
            weightsUpd = obj.updateWeightsNonlinear(particlesPred, state.weights, zNow);

            est = particlesPred * weightsUpd;
            [particlesRes, weightsRes] = obj.resampleEss(particlesPred, weightsUpd);

            state.velPrev = est * ones(1, obj.numParticles) - state.particlesPrev;
            state.particlesPrev = particlesRes;
            state.weights = weightsRes;
            state.estimatedPos(:, pointIdx) = est;
        end

        function particles = sampleToa(obj, center)
            if isempty(obj.toaNoise)
                particles = center + obj.noiseScale * randn(2, obj.numParticles);
                return;
            end

            indices = randi(size(obj.toaNoise, 2), [1, obj.numParticles]);
            particles = center + obj.toaNoise(:, indices);
        end

        function noise = sampleProcess(obj)
            if isempty(obj.processNoise)
                noise = obj.noiseScale * randn(2, obj.numParticles);
                return;
            end

            indices = randi(size(obj.processNoise, 2), [1, obj.numParticles]);
            noise = obj.processNoise(:, indices);
        end

        function y_pred = H_nonlinear(obj, x)
            % Nonlinear observation model: compute ranging to each anchor
            % x: 2 x N (particle positions)
            % y_pred: 4 x N (predicted ranging measurements)
            y_pred = zeros(size(obj.anchorPos, 1), obj.numParticles);
            for i = 1:size(obj.anchorPos, 1)
                dx = x(1, :) - obj.anchorPos(i, 1);
                dy = x(2, :) - obj.anchorPos(i, 2);
                y_pred(i, :) = sqrt(dx.^2 + dy.^2);
            end
        end

        function weights = updateWeightsNonlinear(obj, particles, prevWeights, zNow)
            % Nonlinear update using ranging measurements (ToA)
            % particles: 2 x N (particle states)
            % prevWeights: N x 1 (particle weights)
            % zNow: 4 x 1 (ranging measurements to 4 anchors)
            % weights: N x 1 (updated weights)
            
            y_pred = obj.H_nonlinear(particles);  % 4 x N predicted rangings
            numAnchors = size(y_pred, 1);
            Rinv = eye(numAnchors) / (obj.noiseScale^2) / 3;  % Inverse of R matrix
            errors = zNow - y_pred;  % 4 x N measurement residuals
            
            % Mahalanobis distance: sum((R^-1 * errors) .* errors, 1)
            distances = sum((Rinv * errors) .* errors, 1);  % 1 x N
            
            weights = prevWeights(:)' .* exp(-0.5 * distances);
            weights = weights + 1e-300;
            weights = (weights / sum(weights)).';
        end

        function [particlesOut, weightsOut] = resampleEss(obj, particles, weights)
            ess = 1 / sum(weights .^ 2);
            if ess < obj.numParticles / 2
                wtc = cumsum(weights);
                rpt = rand(obj.numParticles, 1);
                [~, ind1] = sort([rpt; wtc]);
                ind = find(ind1 <= obj.numParticles) - (0:obj.numParticles-1)';
                particlesOut = particles(:, ind);
                weightsOut = ones(obj.numParticles, 1) / obj.numParticles;
            else
                particlesOut = particles;
                weightsOut = weights;
            end
        end
    end
end

function noiseBank = localExtractNoiseBank(noiseTensor, noiseIdx)
    dims = ndims(noiseTensor);

    if dims == 2
        noiseBank = noiseTensor;
        return;
    end

    if dims == 3
        noiseBank = squeeze(noiseTensor(:, :, noiseIdx));
        return;
    end

    if dims == 4
        temp = squeeze(noiseTensor(:, :, :, noiseIdx));
        noiseBank = reshape(temp, size(temp, 1), []);
        return;
    end

    error('PF:UnsupportedNoiseTensor', 'Unsupported noise tensor dimensions: %d', dims);
end
