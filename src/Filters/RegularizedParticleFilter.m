classdef RegularizedParticleFilter < NonlinearParticleFilter
    % RegularizedParticleFilter
    %
    % RPF replaces discrete resampling with kernel-regularized resampling:
    %   1) Draw ancestors from weighted particles
    %   2) Estimate unbiased empirical covariance S (denominator N-1)
    %   3) Compute A by robust Cholesky where A*A' = S
    %   4) Jitter ancestors using Epanechnikov kernel samples

    properties
        bandwidthFloor (1,1) double = 1e-3
    end

    methods
        function obj = RegularizedParticleFilter(data, config, noiseIdx)
            obj@NonlinearParticleFilter(data, config, noiseIdx);

            if isfield(config, 'rpfBandwidthFloor')
                obj.bandwidthFloor = max(config.rpfBandwidthFloor, 0);
            end
        end

        function [state, est] = step(obj, state, iterIdx, pointIdx)
            particlesPred = state.particlesPrev + state.velPrev + obj.processBias + obj.sampleProcess();

            zNow = obj.z(:, pointIdx, iterIdx);
            weightsUpd = obj.updateWeightsNonlinear(particlesPred, state.weights, zNow);

            est = particlesPred * weightsUpd;
            [particlesRes, weightsRes, idxResampled, didResample] = obj.regularizedResampleEss(particlesPred, weightsUpd);

            if didResample
                state.velPrev = particlesRes - state.particlesPrev(:, idxResampled);
            else
                state.velPrev = particlesRes - state.particlesPrev;
            end
            state.particlesPrev = particlesRes;
            state.weights = weightsRes;
            state.estimatedPos(:, pointIdx) = est;
        end

        function [particlesOut, weightsOut, idx, didResample] = regularizedResampleEss(obj, particles, weights)
            ess = 1 / sum(weights .^ 2);
            if ess >= obj.numParticles * obj.resampleThresholdRatio
                particlesOut = particles;
                weightsOut = weights;
                idx = (1:obj.numParticles)';
                didResample = false;
                return;
            end

            idx = obj.systematicResampleIndices(weights, obj.numParticles);
            baseParticles = particles(:, idx);

            [A, h] = obj.computeCovarianceRootAndBandwidth(baseParticles);
            kernelNoise = obj.sampleEpanechnikovNoise(size(baseParticles, 1), obj.numParticles);

            particlesOut = baseParticles + h * (A * kernelNoise);
            weightsOut = ones(obj.numParticles, 1) / obj.numParticles;
            didResample = true;
        end

        function [A, h] = computeCovarianceRootAndBandwidth(obj, particles)
            n = size(particles, 1);
            N = size(particles, 2);

            if N <= 1
                S = (obj.noiseScale^2) * eye(n);
            else
                mu = mean(particles, 2);
                centered = particles - mu;
                % Unbiased empirical covariance with denominator N-1.
                S = (centered * centered.') / (N - 1);
            end

            S = 0.5 * (S + S.');
            A = obj.robustCholesky(S);

            v_n = pi^(n / 2) / gamma(n / 2 + 1); % Unit hypersphere volume.
            h = 0.5 * ((8 * (v_n ^ -1) * (n + 4) * (2 * pi)^n) ^ (1 / (n + 4))) * (N ^ (-1 / (n + 4)));
            h = max(h, obj.bandwidthFloor);
        end

        function A = robustCholesky(~, S)
            n = size(S, 1);
            I = eye(n);
            traceScale = trace(S) / max(n, 1);
            baseJitter = max(1e-12, 1e-10 * max(traceScale, 1));

            Ssym = 0.5 * (S + S.');
            [A, p] = chol(Ssym, 'lower');
            if p == 0
                return;
            end

            for k = 0:6
                jitter = baseJitter * (10 ^ k);
                [A, p] = chol(Ssym + jitter * I, 'lower');
                if p == 0
                    return;
                end
            end

            % Final fallback if S is near-singular/non-PSD.
            [V, D] = eig(Ssym);
            d = diag(D);
            d = max(d, 1e-12);
            A = V * diag(sqrt(d));
        end

        function idx = systematicResampleIndices(~, weights, N)
            cdf = cumsum(weights(:));
            cdf(end) = 1.0;

            u0 = rand / N;
            u = u0 + (0:N-1)' / N;

            idx = zeros(N, 1);
            j = 1;
            for i = 1:N
                while u(i) > cdf(j)
                    j = j + 1;
                end
                idx(i) = j;
            end
        end

        function noise = sampleEpanechnikovNoise(~, n, N)
            noise = zeros(n, N);

            for k = 1:N
                accepted = false;
                while ~accepted
                    dir = randn(n, 1);
                    dirNorm = norm(dir);
                    if dirNorm < 1e-12
                        continue;
                    end

                    dir = dir / dirNorm;
                    r = rand ^ (1 / n); % Radius for proposal uniform-in-ball.
                    candidate = r * dir;

                    % Accept-reject for Epanechnikov profile: K(u) proportional to (1-||u||^2).
                    if rand <= (1 - r^2)
                        noise(:, k) = candidate;
                        accepted = true;
                    end
                end
            end
        end

    end
end