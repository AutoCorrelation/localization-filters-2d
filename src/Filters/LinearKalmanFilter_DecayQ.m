classdef LinearKalmanFilter_DecayQ < LinearKalmanFilter
    properties
        decayGamma
    end

    methods
        function obj = LinearKalmanFilter_DecayQ(data, config, noiseIdx)
            obj@LinearKalmanFilter(data, config, noiseIdx);
            obj.decayGamma = config.decayGamma(noiseIdx);
        end

        function [state, est] = step(obj, state, iterIdx, pointIdx)
            xPrev = state.estimatedPos(:, pointIdx - 1);
            PPrev = state.errCov(:, :, pointIdx - 1);
            velPrev = state.vel(:, pointIdx - 1);

            xPred = xPrev + velPrev + obj.processBias;
            PPred = PPrev + obj.Q * exp(-obj.decayGamma * (pointIdx - 3));

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