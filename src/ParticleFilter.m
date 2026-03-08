classdef ParticleFilter
    properties
        processNoise
        toaNoise
        numParticles
        opti_w_gamma
        Q
        noiseInd
        noiseVar = [0.01 0.1 1 10 100]
        processBias
        anchorPos = [0, 10; 0, 0; 10, 0; 10, 10]  % Match Env.m anchor positions: [0 0 10 10; 10 0 0 10]'
    end

    methods
        function obj = ParticleFilter(Noise, numParticles)
            obj.numParticles = numParticles;
            obj.noiseInd = Noise;
            obj.processNoise = load(['../data/processNoise',num2str(Noise),'.csv']);
            obj.toaNoise = load(['../data/toaNoise',num2str(Noise),'.csv']);
            obj.Q = load(['../data/Q', num2str(Noise), '.csv']);
            obj.processBias = load(['../data/processBias', num2str(Noise), '.csv']);
        end



        function y = sampling(obj, x)
            % Vectorized: sample noise indices and add to state
            indices = ceil(size(obj.toaNoise, 2) * rand(1, obj.numParticles));
            y = x + obj.toaNoise(:, indices);
            % y = x + sqrt(obj.noiseVar(obj.noiseInd)) * randn(2, obj.numParticles);
        end

        function y = predict(obj, x, B, u)
            % Vectorized: predict with velocity and process noise
            indices = ceil(size(obj.processNoise, 2) * rand(1, obj.numParticles));
            noise = obj.processNoise(:, indices);
            % y = x + B .* u + noise + obj.processBias;
            y = x + B .* u + noise;
        end

        function y = predictParam(obj, x, B, u, countStep, gamma)
            % Vectorized: predict with decaying process noise
            indices = ceil(size(obj.processNoise, 2) * rand(1, obj.numParticles));
            noise = obj.processNoise(:, indices);
            decay = exp(-gamma * (countStep - 2));
            y = x + B .* u + noise * decay;
            % y = x + B .* u + noise * decay + obj.processBias;
        end

        function y = update(~, x, w, z, H, R)
            % Vectorized: compute likelihood for all particles
            R = R + 1e-8 * eye(size(R));
            errors = z - H * x;  % (M x N) matrix
            % Solve R * v = errors directly instead of computing the full inverse
            Rinv_errors = R \ errors;  % More efficient and numerically stable than R\eye*errors
            distances = sum(errors .* Rinv_errors, 1);  % Element-wise squared Mahalanobis
            y = (w(:)' .* exp(-0.5 * distances)) + 1e-300;  % Avoid zero weights
            y = y / sum(y);
            y = y(:);  % Ensure column vector output
        end

        function weight = updateNonLinear(obj, x, w, z)
            % Non-linear update using ranging measurements (ToA)
            % x: 2 x N (particle states)
            % w: N x 1 (particle weights)
            % z: 4 x 1 (ranging measurements to 4 anchors)
            % weight: N x 1 (updated weights)
            
            y_pred = obj.H_nonlinear(x);  % 4 x N predicted rangings
            errors = z - y_pred;  % 4 x N measurement residuals
            
            % R is a scaled identity: (3 * noiseVar) * I_4, so R^{-1} scales by 1/(3*noiseVar)
            inv_scale = 1 / (3 * obj.noiseVar(obj.noiseInd));
            distances = inv_scale * sum(errors.^2, 1);  % 1 x N
            
            weight = w(:) .* exp(-0.5 * distances') + 1e-300;  % N x 1
            weight = weight / sum(weight);  % Normalize
            weight = weight(:);  % Ensure column vector output
        end

        function y = H_nonlinear(obj, x)
            % x: 2 x N, anchorPos: 4 x 2
            % Use implicit broadcasting: anchorPos(:,1) is 4x1, x(1,:) is 1xN -> 4xN
            dx = x(1, :) - obj.anchorPos(:, 1);  % 4 x N
            dy = x(2, :) - obj.anchorPos(:, 2);  % 4 x N
            y = sqrt(dx.^2 + dy.^2);
        end

        function y = estimate(~, x, w)
            % Vectorized: weighted sum of particles
            % Ensure w is column vector
            y = x * w(:);
        end

        function [y,weight] = resamplingEss(obj, x, w)
            % Vectorized: ESS computation
            Npt = length(w);
            Ess = 1 / sum(w.^2);
            if Ess < Npt/2
                wtc = cumsum(w);
                rpt = rand(Npt, 1);
                [~, ind1] = sort([rpt; wtc]);
                ind = find(ind1 <= Npt) - (0:Npt-1)';
                y = x(:, ind);
                y = obj.roughening(y, 0.2); % roughening only after resampling
                weight = ones(obj.numParticles, 1) / obj.numParticles;
            else
                y = x;
                weight = w;
            end
        end

        function [y,weight] = resampling(obj, x, w)
            Npt = length(w);
            wtc = cumsum(w);
            rpt = rand(Npt, 1);
            [~, ind1] = sort([rpt; wtc]);
            ind = find(ind1 <= Npt) - (0:Npt-1)';
            y = x(:, ind);
            % y = obj.roughening(y, 0.2); % roughening only after resampling
            weight = ones(obj.numParticles, 1) / obj.numParticles;
        end


        function y = roughening(obj, x, K)
            % Vectorized: add zero-mean Gaussian roughening to particles
            % K: tuning constant; N: numParticles; dx: state dimension
            N = obj.numParticles;
            dx = 2;
            D = max(abs(diff(x, 1, 2)), [], 2);
            sigma = K * D * N^(-1/dx);
            y = x + sigma .* randn(size(x));
        end

        function y = updateParam(~, x, w, z, pinvH, R, gamma)
            % Vectorized: update with multivariate normal likelihood across all particles
            R = R + 1e-6 * eye(size(R));
            Rcov = R * gamma;
            % Compute predicted observations and residuals for all particles at once
            residuals = z(:) - pinvH * x;  % 6 x N
            % Solve Rcov * v = residuals (avoids explicit matrix inverse)
            Rcov_inv_residuals = Rcov \ residuals;  % 6 x N
            log_likelihoods = -0.5 * sum(residuals .* Rcov_inv_residuals, 1);  % 1 x N
            % Subtract maximum for numerical stability before exponentiation
            log_likelihoods = log_likelihoods - max(log_likelihoods);
            y = w(:)' .* exp(log_likelihoods);
            y = y / sum(y);
        end

        function [y,weight] = resampling_param(obj, x, w, ~, gamma)
            % Vectorized: ESS computation
            Npt = length(w);
            Ess = 1 / sum(w.^2);
            if Ess < Npt*gamma % scalar mode
                % if Ess < Npt*(exp(gamma*(countStep-2))) % increase mode
                wtc = cumsum(w);
                rpt = rand(Npt, 1);
                [~, ind1] = sort([rpt; wtc]);
                ind = find(ind1 <= Npt) - (0:Npt-1)';
                y = x(:, ind);
                y = obj.roughening(y, 0.2); % roughening only after resampling
                weight = ones(obj.numParticles, 1) / obj.numParticles;
            else
                y = x;
                weight = w;
            end
        end

        function y = metropolis_resampling(~, x, w)
            % Metropolis-Hastings resampling
            N = length(w);
            Ess = 1 / sum(w.^2);
            if Ess < N/2
                indices = zeros(1, N);
                indices(1) = randi(N);
                for i = 2:N
                    candidate = randi(N);
                    acceptance_ratio = w(candidate) / w(indices(i-1));
                    if rand <= acceptance_ratio
                        indices(i) = candidate;
                    else
                        indices(i) = indices(i-1);
                    end
                end
                y = x(:,indices);
            else
                y = x;
            end
        end

        function [y, weight] = multinomial_resampling(obj, x, w)
            % Vectorized: multinomial resampling
            N = length(w);
            Ess = 1 / sum(w.^2);
            if Ess < N/2
                edges = min([0 cumsum(w')], 1); % Cumulative sum of weights
                edges(end) = 1; % Ensure sum is exactly one
                u = rand(N, 1); % Multinomial: independent uniform samples
                [~, ~, indices] = histcounts(u, edges); % Find indices
                y = x(:, indices); % Resample particles
                y = obj.roughening(y, 0.2); % roughening only after resampling
                weight = ones(obj.numParticles, 1) / obj.numParticles;
            else
                y = x;
                weight = w;
            end
        end

        function y = systematic_resampling(~, x, w)
            % Systematic resampling
            N = length(w);
            Ess = 1 / sum(w.^2);
            if Ess < N/2
                positions = (rand + (0:N-1)) / N;
                c = cumsum(w(:));
                c(end) = 1;  % Guard against floating-point rounding
                indexes = discretize(positions, [0; c]);
                y = x(:, indexes);
            else
                y = x;
            end
        end

        function y = stratified_resampling(~, x, w)
            % Stratified resampling
            N = length(w);
            Ess = 1 / sum(w.^2);
            if Ess < N/2
                positions = ((0:N-1) + rand(1, N)) / N;
                c = cumsum(w(:));
                c(end) = 1;  % Guard against floating-point rounding
                indexes = discretize(positions, [0; c]);
                y = x(:, indexes);
            else
                y = x;
            end
        end

        function y = residual_resampling(~, x, w)
            % Residual resampling
            N = length(w);
            Ess = 1 / sum(w.^2);
            if Ess < N/2

                indexes = zeros(1, N);
                residuals = N * w - floor(N * w);
                indexes(1:sum(floor(N * w))) =  repelem(1:N, floor(N * w));
                cumulative_sum = cumsum(residuals);
                positions = (rand + (0:N-1)') / N;
                i = 1;
                for j = sum(floor(N * w))+1:N
                    while positions(j) > cumulative_sum(i)
                        i = i + 1;
                    end
                    indexes(j) = i;
                end
                y = x(:, indexes); % Resample particles
            else
                y = x;
            end
        end


        function obj = thresholding(obj, Noise)
            % Threshold process noise magnitudes
            Qload = obj.Q;
            Qmax = Qload(1);
            norms = sqrt(sum(obj.processNoise.^2, 1));

            % Alternating sign pattern (+1, -1, +1, ...) for all particles
            sign_pattern = repmat([1; -1], 1, ceil(obj.numParticles/2));
            sign_pattern = sign_pattern(1:obj.numParticles);

            % Replace high-norm particles using logical indexing (no loop)
            mask = norms > Qmax;
            obj.processNoise(:, mask) = [1; -1] * (sqrt(Qmax) .* sign_pattern(mask));
        end

        function [m, s] = belief(~, z, H, xparticles, m, s)
            % Global AdaBelief moments calculation for weighted estimate
            % xparticles: 2 x 1 (weighted particle estimate)
            % z: 6 x 1 (measurement vector)
            % H: 6 x 2 (observation matrix)
            % m, s: 6 x 1 (1st and 2nd moments of innovation)
            
            beta1 = 0.9;
            beta2 = 0.999;
            
            % Innovation: measurement residual
            g = z - H * xparticles; % 6 x 1
            
            % Update moments with exponential moving average
            m = beta1 * m + (1 - beta1) * g;
            s = beta2 * s + (1 - beta2) * ((g - m).^2) + 1e-8;
            
            % Bias correction
            m = m / (1 - beta1);
            s = s / (1 - beta2);
        end


    end
end

