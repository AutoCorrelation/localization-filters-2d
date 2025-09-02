classdef ParticleFilter
    properties
        processNoise
        toaNoise
        numParticles
    end

    methods
        function obj = ParticleFilter(Noise, numParticles)
            obj.numParticles = numParticles;
            obj.processNoise = load(['../data/processNoise',num2str(Noise),'.csv']);
            obj.toaNoise = load(['../data/toaNoise',num2str(Noise),'.csv']);
        end

        function obj = thresholding(obj,Noise)
            Q = load(['../data/Q',num2str(Noise),'.csv']);
            Qmax = Q(1);
            hanroro = sqrt(Qmax);
            for i=1:obj.numParticles
                if Qmax<=norm(obj.processNoise(:,i))
                    obj.processNoise(:,i) = [hanroro;-hanroro];
                    hanroro = -hanroro;
                end
            end
        end

        function y = sampling(obj, x) % me: toa+toaNoise / SHyeon: toa+based on wk, Q
            y = zeros(2, obj.numParticles);
            for k = 1:obj.numParticles
                index = ceil(size(obj.toaNoise, 2) * rand);
                
                y(:, k) = x + obj.toaNoise(:, index);
                % y(:, k) = x + obj.toaNoise(:, k);
            end
        end

        function y = predict(obj, x, B, u)
            y = zeros(size(x));
            for k = 1:obj.numParticles
                index = ceil(size(obj.processNoise, 2) * rand);
                noise = obj.processNoise(:, index);
                y(:, k) = x(:, k) + B(:, k) * u + noise;
            end
        end


        function y = update(~, x, w, z, pinvH, R)
            y = zeros(size(w));
            R = R + 1e-6 * eye(size(R));
            for k = 1:length(w)
                % Method 1
                % y(k) = w(k) * mvnpdf(z, pinvH*x(:, k), R);

                % Method 2
                error = z - pinvH * x(:, k);
                y(k) = w(k) * exp(-0.5 * (error' * (R \ error)));

                % Method 3
                % y(k) = w(k) * mvnpdf(pinvH * z, x(:, k), pinvH * R * pinvH');
            end
            y = y / sum(y);
        end



        function y = estimate(~, x, w)
            y = 0;
            for k = 1:length(w)
                y = y + w(k) * x(:, k);
            end
        end

        function y = resample(~, x, w)
            var_accum = 0;
            Npt = length(w);
            for ind = 1:Npt
                var_accum = var_accum + w(ind)^2;
            end
            Ess = 1 / var_accum;
            if Ess < Npt*2/3
                wtc = cumsum(w);
                rpt = rand(Npt, 1);
                [~, ind1] = sort([rpt; wtc]);
                ind = find(ind1 <= Npt) - (0:Npt-1)';
                y = x(:, ind);
            else
                y = x;
            end
        end


        function y = roughening(obj,x, K)
            % K is positive tuning constant
            % N number of particles
            % D is max(difference of components)
            % dx dimension of state  % sigma = KDN^(-1/dx)
            N = obj.numParticles;
            dx = 2;
            D = max(abs(diff(x, 1, 2)), [], 2);
            sigma = K * D * N^(-1/dx);
            for j = 1:N
                x(:, j) = x(:, j) + sigma .* randn(dx, 1);
            end
            y = x;
        end


        function y = predictParam(obj, x, B, u, countStep, gamma)
            y = zeros(size(x));
            for k = 1:obj.numParticles
                index = ceil(size(obj.processNoise, 2) * rand);
                noise = obj.processNoise(:, index);
                % index = k;
                y(:, k) = x(:, k) + B(:, k) * u + noise * exp(-gamma*(countStep-2));
                % y(:, k) = x(:, k) + B(:, k) * u + noise * gamma^(countStep-2);
            end
        end

        function y = updateParam(~, x, w, z, pinvH, R, gamma)
            y = zeros(size(w));
            R = R + 1e-6 * eye(size(R));
            for k = 1:length(w)
                y(k) = w(k) * mvnpdf(z, pinvH*x(:, k), R*gamma);
                % y(k) = w(k) * mvnpdf(pinvH * z, x(:, k), pinvH * R * pinvH');
            end
            y = y / sum(y);
        end

        function y = metropolis_resampling(~,x,w)
            N = length(w);
            var_accum = 0;
            Npt = length(w);
            for ind = 1:Npt
                var_accum = var_accum + w(ind)^2;
            end
            Ess = 1 / var_accum;
            if Ess < Npt*2/3
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

        function y = multinomial_resampling(~,x,w)
            N = length(w);
            var_accum = 0;
            Npt = length(w);
            for ind = 1:Npt
                var_accum = var_accum + w(ind)^2;
            end
            Ess = 1 / var_accum;
            if Ess < Npt*2/3
                edges = min([0 cumsum(w')], 1); % Cumulative sum of weights
                edges(end) = 1; % Ensure sum is exactly one
                u1 = rand/N; % Start of uniform distribution
                u = u1 + (0:N-1)'/N; % Uniform distribution
                [~, ~, indices] = histcounts(u, edges); % Find indices
                y = x(:, indices); % Resample particles
            else
                y = x;
            end
        end

        function y = systematic_resampling(~,x, w)
            N = length(w);
            var_accum = 0;
            Npt = length(w);
            for ind = 1:Npt
                var_accum = var_accum + w(ind)^2;
            end
            Ess = 1 / var_accum;
            if Ess < Npt*2/3
                positions = (rand + (0:N-1)) / N;
                indexes = zeros(1, N);
                cumulative_sum = cumsum(w);
                i = 1;
                for j = 1:N
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

        function y = stratified_resampling(~,x,w)
            N = length(w);
            var_accum = 0;
            Npt = length(w);
            for ind = 1:Npt
                var_accum = var_accum + w(ind)^2;
            end
            Ess = 1 / var_accum;
            if Ess < Npt*2/3
                positions = ((0:N-1) + rand(1, N)) / N;
                indexes = zeros(1, N);
                cumulative_sum = cumsum(w);
                i = 1;
                for j = 1:N
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

        function y = residual_resampling(~,x,w)
            N = length(w);
            var_accum = 0;
            Npt = length(w);
            for ind = 1:Npt
                var_accum = var_accum + w(ind)^2;
            end
            Ess = 1 / var_accum;
            if Ess < Npt*2/3

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
    end
end
