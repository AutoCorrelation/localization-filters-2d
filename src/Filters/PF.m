classdef PF
    properties
        H
        xHat
        z
        R
        processNoise
        toaNoise
        numParticles
        noiseScale
    end

    methods
        function obj = PF(data, config, noiseIdx)
            obj.H = config.H;
            obj.xHat = squeeze(data.x_hat_LLS(:, :, :, noiseIdx));
            obj.z = squeeze(data.z_LLS(:, :, :, noiseIdx));
            obj.R = squeeze(data.R_LLS(:, :, :, :, noiseIdx));

            obj.processNoise = localExtractNoiseBank(data.processNoise, noiseIdx);
            obj.toaNoise = localExtractNoiseBank(data.toaNoise, noiseIdx);

            if isfield(config, 'numParticles')
                obj.numParticles = config.numParticles;
            else
                obj.numParticles = 500;
            end

            obj.noiseScale = sqrt(config.noiseVariance(noiseIdx));
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

            Rmat = obj.R(:, :, pointIdx, iterIdx);
            zNow = obj.z(:, pointIdx, iterIdx);
            weightsUpd = obj.updateWeightsLinear(particlesPred, state.weights, zNow, Rmat);

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

        function weights = updateWeightsLinear(obj, particles, prevWeights, zNow, Rmat)
            Rmat = Rmat + 1e-8 * eye(size(Rmat));
            residual = zNow - obj.H * particles;
            Rinv = Rmat \ eye(size(Rmat));
            dist = sum((Rinv * residual) .* residual, 1);

            weights = prevWeights(:)' .* exp(-0.5 * dist);
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
