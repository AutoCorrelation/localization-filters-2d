classdef MCMCResamplingParticleFilter < NonlinearParticleFilter
    % MCMCResamplingParticleFilter
    %
    % Nonlinear PF with Metropolis-Hastings based resampling:
    %   1) Perform systematic resampling based on weights
    %   2) Apply MH steps to mix particles while preserving likelihood structure
    %   3) Acceptance probability checks if new state has better likelihood
    %
    % [Metropolis-Hastings Acceptance Probability]
    %   α = min[1, (p(y_k|x_k,i^-) * p(x_k,i^-|x_{k-1,i}^+)) / (p(y_k|x~_k,i) * p(x~_k,i|x_{k-1,i}^+))]
    %
    %   where:
    %     x_k,i^- : previous particle state
    %     x~_k,i  : proposed particle state
    %     y_k     : current measurement
    %     p(y_k|x) : observation likelihood
    %     p(x|x_prev) : process model prior

    properties
        % Number of MH iterations per resampling cycle
        mcmcIter     (1,1) double = 5

        % Number of MH steps per particle per iteration
        mcmcStepsPerIter (1,1) double = 1

        % Proposal noise scale (relative to process noise)
        proposalScale (1,1) double = 1.0
    end

    methods
        function obj = MCMCResamplingParticleFilter(data, config, noiseIdx)
            obj@NonlinearParticleFilter(data, config, noiseIdx);

            if isfield(config, 'mcmcIter')
                obj.mcmcIter = max(config.mcmcIter, 1);
            end
            if isfield(config, 'mcmcStepsPerIter')
                obj.mcmcStepsPerIter = max(config.mcmcStepsPerIter, 1);
            end
            if isfield(config, 'proposalScale')
                obj.proposalScale = max(config.proposalScale, 0.1);
            end
        end

        function [state, est] = step(obj, state, iterIdx, pointIdx)
            % Predict particles
            particlesPred = state.particlesPrev + state.velPrev + obj.processBias + obj.sampleProcess();

            % Measure and update weights
            zNow = obj.z(:, pointIdx, iterIdx);
            weightsUpd = obj.updateWeightsNonlinear(particlesPred, state.weights, zNow);

            % Estimate position
            est = particlesPred * weightsUpd;

            % MCMC-based resampling
            [particlesRes, weightsRes, idxAnc] = obj.mcmcResampleEss( ...
                particlesPred, state.particlesPrev, state.velPrev, weightsUpd, zNow);

            % Update state
            state.velPrev = particlesRes - state.particlesPrev(:, idxAnc);
            state.particlesPrev = particlesRes;
            state.weights = weightsRes;
            state.estimatedPos(:, pointIdx) = est;
        end

        function [particlesOut, weightsOut, idxAnc] = mcmcResampleEss(obj, particles, prevParticles, prevVel, weights, zNow)
            % Check if resampling is needed
            ess = 1 / sum(weights .^ 2);
            if ess >= obj.numParticles * obj.resampleThresholdRatio
                particlesOut = particles;
                weightsOut = weights;
                idxAnc = (1:obj.numParticles)';
                return;
            end

            % Systematic resampling to get parent indices
            idx = obj.systematicResampleIndices(weights, obj.numParticles);
            particlesBase = particles(:, idx);
            muAnc = prevParticles(:, idx) + prevVel(:, idx) + obj.processBias;
            idxAnc = idx;

            % Apply MH iterations for mixing
            particlesMH = particlesBase;
            acceptCount = 0;
            totalAttempts = 0;

            for iter = 1:obj.mcmcIter
                for step = 1:obj.mcmcStepsPerIter
                    for i = 1:obj.numParticles
                        % Proposal: random walk with scaled Gaussian noise
                        proposalNoise = obj.proposalScale * obj.noiseScale * randn(2, 1);
                        xProposal = particlesMH(:, i) + proposalNoise;

                        % Compute likelihoods for current and proposed states
                        yPredCurrent = obj.H_nonlinear(particlesMH(:, i));
                        yPredProposal = obj.H_nonlinear(xProposal);

                        llhCurrent = obj.computeLogLikelihood(zNow, yPredCurrent);
                        llhProposal = obj.computeLogLikelihood(zNow, yPredProposal);

                        % Transition prior term for MH target:
                        % pi(x) ∝ p(z_k|x) * p(x|x_{k-1,ancestor})
                        lpCurrent = obj.computeLogTransitionPrior(particlesMH(:, i), muAnc(:, i));
                        lpProposal = obj.computeLogTransitionPrior(xProposal, muAnc(:, i));

                        % MH Acceptance Probability
                        % α = min[1, exp((log p(y|x_new)+log p(x_new|x_prev))
                        %                - (log p(y|x_old)+log p(x_old|x_prev)))]
                        % Symmetric RW proposal cancels q-ratio.
                        alpha = min(1, exp((llhProposal + lpProposal) - (llhCurrent + lpCurrent)));

                        % Accept/Reject Decision
                        if rand() < alpha
                            particlesMH(:, i) = xProposal;
                            acceptCount = acceptCount + 1;
                        end
                        totalAttempts = totalAttempts + 1;
                    end
                end
            end

            particlesOut = particlesMH;
            weightsOut = ones(obj.numParticles, 1) / obj.numParticles;
        end

        function llh = computeLogLikelihood(obj, zNow, yPred)
            % Compute Gaussian log-likelihood: log N(z; y, R)
            %   = -0.5 * ||z - y||_R^{-2}^2 - 0.5*log(det(R)) - n*log(pi)
            % We drop constant terms and use diagonal R
            numAnchors = size(yPred, 1);
            Rinv = eye(numAnchors) / (obj.noiseScale^2);
            errors = zNow - yPred;
            distances = sum((Rinv * errors) .* errors, 1);
            llh = -0.5 * distances;
        end

        function lp = computeLogTransitionPrior(obj, xNow, muTrans)
            % Gaussian transition prior approximation from process model.
            diff = xNow - muTrans;
            invVar = 1 / (obj.noiseScale ^ 2);
            lp = -0.5 * sum((diff .^ 2) * invVar);
        end

        function idx = systematicResampleIndices(~, weights, N)
            % Systematic resampling using CDF
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

    end
end
