classdef GTParticleFilter
    properties
        processNoise
        toaNoise
        numParticles
        numTransportIters
        epsilonScale
        epsilonMax
        minDist
        regEps
        gamma
        betaSchedule
    end

    methods
        function obj = GTParticleFilter(Noise, numParticles, numTransportIters, epsilonScale, gamma, betaSchedule)
            if nargin < 3
                numTransportIters = 10;
            end
            if nargin < 4
                epsilonScale = 1.0;
            end
            if nargin < 5
                gamma = 0.6;
            end
            if nargin < 6 || isempty(betaSchedule)
                betaSchedule = linspace(0.25, 1.0, numTransportIters);
            end

            obj.numParticles = numParticles;
            obj.numTransportIters = numTransportIters;
            obj.epsilonScale = epsilonScale;
            obj.epsilonMax = 0.5;
            obj.minDist = 1e-6;
            obj.regEps = 1e-8;
            obj.gamma = gamma;
            obj.betaSchedule = betaSchedule(:);

            obj.processNoise = load(['../data/processNoise', num2str(Noise), '.csv']);
            obj.toaNoise = load(['../data/toaNoise', num2str(Noise), '.csv']);
        end

        function y = sampling(obj, x)
            y = zeros(2, obj.numParticles);
            for k = 1:obj.numParticles
                index = ceil(size(obj.toaNoise, 2) * rand);
                y(:, k) = x + obj.toaNoise(:, index);
            end
        end

        function y = predict(obj, x, B, u, countStep)
            y = zeros(size(x));
            decay = exp(-obj.gamma * max(countStep - 2, 0));
            for k = 1:obj.numParticles
                index = ceil(size(obj.processNoise, 2) * rand);
                noise = obj.processNoise(:, index);
                y(:, k) = x(:, k) + B(:, k) * u + decay * noise;
            end
        end

        function [particles, info] = transport(obj, particles, anchors, z, R)
            M = size(anchors, 2);
            K = obj.numTransportIters;
            info.meanGradNorm = zeros(K, 1);
            info.meanStep = zeros(K, 1);

            for k = 1:K
                beta = obj.betaSchedule(min(k, length(obj.betaSchedule)));
                beta = max(beta, 1e-6);
                Rk = (R + obj.regEps * eye(size(R))) / beta;

                gradNormAccum = 0;
                stepAccum = 0;

                for i = 1:obj.numParticles
                    x = particles(:, i);
                    h = zeros(M, 1);
                    H = zeros(M, 2);

                    for m = 1:M
                        diffVec = x - anchors(:, m);
                        dist = norm(diffVec);
                        if dist < obj.minDist
                            dist = obj.minDist;
                        end
                        h(m) = dist;
                        H(m, :) = (diffVec / dist)';
                    end

                    grad = H' * (Rk \ (z - h));
                    G = H' * (Rk \ H);
                    G = (G + G') / 2;
                    lambdaMax = max(real(eig(G)));
                    if lambdaMax < obj.regEps
                        lambdaMax = obj.regEps;
                    end

                    epsilon = min(obj.epsilonMax, obj.epsilonScale / lambdaMax);
                    particles(:, i) = x + epsilon * grad;

                    gradNormAccum = gradNormAccum + norm(grad);
                    stepAccum = stepAccum + epsilon;
                end

                info.meanGradNorm(k) = gradNormAccum / obj.numParticles;
                info.meanStep(k) = stepAccum / obj.numParticles;
            end
        end

        function y = estimate(~, particles)
            y = mean(particles, 2);
        end
    end
end
