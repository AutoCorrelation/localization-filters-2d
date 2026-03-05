function [all_gap_loc_PF] = PF(dt, iter, Npoints, Npt, exactPos, Q, w_k, NoiseVar, ToA_X, ToA_Y)
    run_errors = zeros(iter, 1);
    
    if size(w_k, 1) == 2
        w_k = w_k';
    end

    for run = 1:iter
        % 1. 초기화: 첫 ToA 지점 근처에 파티클 생성
        particles = [ToA_X(run, 1), ToA_Y(run, 1)] + sqrt(NoiseVar) * randn(Npt, 2);
        weights = ones(Npt, 1) / Npt;
        x_est = zeros(Npoints, 2);
        
        for t = 1:Npoints
            % 초기 2스텝 보정 (LLS/ToA 성능 보완 로직)
            if t < 3
                x_est(t, :) = [ToA_X(run, t), ToA_Y(run, t)];
                particles = x_est(t, :) + sqrt(NoiseVar) * randn(Npt, 2);
                weights = ones(Npt, 1) / Npt;
            else
                % Step 1: Prediction (Importance Sampling)
                v = (x_est(t-1, :) - x_est(t-2, :)) / dt;
                processNoise = mvnrnd([0, 0], Q, Npt);
                particles = particles + (v * dt) + w_k + processNoise;

                % Step 2: Weight Update (Likelihood 계산)
                z = [ToA_X(run, t), ToA_Y(run, t)]; 
                for i = 1:Npt
                    diff = z - particles(i, :);
                    weights(i) = exp(-0.5 * (diff * diff') / NoiseVar);
                end
                
                % Step 3: Normalization
                sum_w = sum(weights);
                if sum_w < 1e-300
                    weights = ones(Npt, 1) / Npt;
                else
                    weights = weights / sum_w;
                end
                
                % Step 4: State Estimation
                x_est(t, :) = sum(particles .* weights, 1);

                % Step 5: Resampling
                Neff = 1 / sum(weights.^2);
                if Neff < Npt / 2
                    [particles, weights] = SIR(particles, weights);
                end
            end
        end
        run_errors(run) = norm(x_est - exactPos, 'fro') / Npoints;
    end
    all_gap_loc_PF = mean(run_errors(~isnan(run_errors)));
end