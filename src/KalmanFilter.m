classdef KalmanFilter
    %KALMANFILTER
    %   Kalman Filtering for 2D TOA
    %   This class implements the Kalman filter algorithm for 2D Time of Arrival (TOA) localization.
    %   구조체를 이용해서 다른 변수들도 전달할 수 있도록 하고, 특히 Q를 구조체로 전달할 수 있도록 한다.
    %   properties에는 아무것도 넣지 않음.
    
    properties
        pNoiseCov   % Process noise covariance
        bias        % Bias
        K         % Kalman gain
        H        % Measurement matrix
    end
    
    methods
        function obj = KalmanFilter(Noise, H)
            obj.pNoiseCov = load(['../data/Q', num2str(Noise), '.csv']);
            obj.bias = load(['../data/processbias', num2str(Noise), '.csv']);
            obj.H = H;
        end
        
        function [xhat, Phat] = predict(obj,x, P, B, u)
            xhat = x + B * u + obj.bias;
            Phat = P + obj.pNoiseCov;
        end
        

        function obj = update(obj,P, R)
            R = R + 1e-6 * eye(size(R)); % Add small value to avoid singularity
            obj.K = P * obj.H' / (obj.H * P * obj.H' + R);
        end

        function [x, P] = estimate(obj, x, P, z)
            x = x + obj.K * (z - obj.H * x);
            P = (eye(size(P)) - obj.K * obj.H) * P;
        end
    end
end

