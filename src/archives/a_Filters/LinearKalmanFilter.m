classdef LinearKalmanFilter
    properties
        H
        xHat
        z
        R
        Q
        processBias
        P0
    end

    methods
        function obj = LinearKalmanFilter(data, config, noiseIdx)
            obj.H = config.H;
            obj.xHat = squeeze(data.x_hat_LLS(:, :, :, noiseIdx));
            obj.z = squeeze(data.z_LLS(:, :, :, noiseIdx));
            obj.R = squeeze(data.R_LLS(:, :, :, :, noiseIdx));
            obj.Q = squeeze(data.Q(:, :, noiseIdx));
            obj.P0 = squeeze(data.P0(:, :, noiseIdx));


            processBiasRaw = squeeze(data.processbias(:, noiseIdx));
            obj.processBias = reshape(processBiasRaw, [2, 1]);
        end

        function state = initializeState(~, numPoints)
            state.estimatedPos = zeros(2, numPoints);
            state.errCov = zeros(2, 2, numPoints);
            state.vel = zeros(2, numPoints);
        end

        function [state, p1, p2] = initializeFirstTwo(obj, state, iterIdx)
            p1 = obj.xHat(:, 1, iterIdx);
            p2 = obj.xHat(:, 2, iterIdx);
            
            state.errCov(:, :, 1) = obj.P0;
            state.errCov(:, :, 2) = obj.P0;
            state.estimatedPos(:, 1) = p1;
            state.estimatedPos(:, 2) = p2;
            state.vel(:, 2) = p2 - p1;
        end

        function [state, est] = step(obj, state, iterIdx, pointIdx)
            xPrev = state.estimatedPos(:, pointIdx - 1);
            PPrev = state.errCov(:, :, pointIdx - 1);
            velPrev = state.vel(:, pointIdx - 1);

            xPred = xPrev + velPrev + obj.processBias;
            PPred = PPrev + obj.Q;

            Rmat = obj.R(:, :, pointIdx, iterIdx);
            Rmat = Rmat + 1e-6 * eye(size(Rmat));
            K = PPred * obj.H' / (obj.H * PPred * obj.H' + Rmat);

            zNow = obj.z(:, pointIdx, iterIdx);
            est = xPred + K * (zNow - obj.H * xPred);
            PNow = (eye(2) - K * obj.H) * PPred;

            state.estimatedPos(:, pointIdx) = est;
            state.errCov(:, :, pointIdx) = PNow;
            state.vel(:, pointIdx) = est - xPrev;
        end
    end
end