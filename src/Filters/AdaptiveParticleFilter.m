classdef AdaptiveParticleFilter < NonlinearParticleFilter
    % AdaptiveParticleFilter  AdaBelief ?��???R-inflation ?�티???�터
    %
    % [?�학??배경]
    %   ?��? PF?�서??고정??R = sigma^2 * I �??�도 ?�수???�용?�다.
    %   R???�위?�으�??�창(inflation)?�면 ?�플 빈약??sample impoverishment)??
    %   강인?��?지�? ?�근??최적??asymptotic optimality)???�실?�다.
    %
    %   ???�터??�??�텝?�서 글로벌 ?�차�?AdaBelief 모멘?��? 추적?�고,
    %   R_k = R_nom + lambdaR * diag(s_k) �?관�?공분?�을 ?�창?�킨??
    %
    %     e_k    = z_k - H(x?_k)                                (가�??�균 ?�측 ?�차)
    %     m_k(i) = beta·m_{k-1}(i) + (1-beta)·e_k(i)           (1�?모멘??EMA)
    %     s_k(i) = beta·s_{k-1}(i) + (1-beta)·(e_k(i)-m_k(i))^2 (AdaBelief 2�?모멘??
    %     R_k    = R_nom + lambdaR * diag(s_k)             (?��??�창)
    %
    %   ?�후 가중치 ?�데?�트??R_k�??�용??가?�시???�도�??�행?�다.
    %     w_i ??w_{i-1} · exp(-0.5 · eᵢ�? R_k?��?e�?
    %
    % [참고] beta가 1??가까울?�록 과거 추정치�? 강하�??��?(?�린 ?�응),
    %        0??가까울?�록 ?�재 ?�차??민감?�게 반응(빠른 ?�응).
    %
    % [?�용 ??
    %   filterObj = AdaptiveParticleFilter(data, config, noiseIdx);             % beta=0.99, lambdaR=1.0
    %   filterObj = AdaptiveParticleFilter(data, config, noiseIdx, 0.8, 2.0);  % beta=0.8, lambdaR=2.0

    properties
        % AdaBelief 계수(EMA 망각 ?�자): 0 < beta < 1
        beta    (1,1) double = 0.99

        % R inflation 강도: R_k = R_nom + lambdaR * diag(s_k)
        lambdaR (1,1) double = 1.0

        % R ?��??�소???�한 / ?�한 (?�치 ?�정??보장)
        rFloor  (1,1) double = 1e-6
        rCeil   (1,1) double = 1e4
    end

    methods
        function obj = AdaptiveParticleFilter(data, config, noiseIdx, beta, lambdaR)
            % ?�성??
            %   beta (?�택): AdaBelief 계수, 기본�?0.99
            %   lambdaR (?�택): R inflation 강도, 기본�?1.0
            obj@NonlinearParticleFilter(data, config, noiseIdx);

            if nargin >= 4 && ~isempty(beta)
                obj.beta = beta;
            end
            if nargin >= 5 && ~isempty(lambdaR)
                obj.lambdaR = lambdaR;
            end
        end

        function state = initializeState(obj, numPoints)
            % 부�??�래??state + ?�응??R ?�태 추�?
            state = initializeState@NonlinearParticleFilter(obj, numPoints);

            % 명목 R ?��?�?2�?모멘???�태 초기??
            numAnchors = size(obj.anchorPos, 1);
            nominalVar = obj.noiseScale^2;                    % noiseVariance(noiseIdx)
            state.nominalDiagR = nominalVar * ones(numAnchors, 1);
            state.mMoment = zeros(numAnchors, 1);
            state.sMoment = zeros(numAnchors, 1);
            state.diagR = state.nominalDiagR;
        end

        function [state, est] = step(obj, state, iterIdx, pointIdx)
            % -----------------------------------------------------------
            % 1. ?�태 ?�측 (부모�? ?�일)
            % -----------------------------------------------------------
            particlesPred = state.particlesPrev + state.velPrev + obj.processBias + obj.sampleProcess();

            % -----------------------------------------------------------
            % 2. 측정�?
            % -----------------------------------------------------------
            zNow = obj.z(:, pointIdx, iterIdx);

            % -----------------------------------------------------------
            % 3. 가�??�균 ?�티?�로 ?�차 계산
            %      x?_k = Σ w_i · x_i
            %      ŷ_k = H(x?_k)  ?? e_k = z_k - ŷ_k
            % -----------------------------------------------------------
            xHatWeighted  = particlesPred * state.weights;            % 2 × 1
            yPredWeighted = obj.H_nonlinear(xHatWeighted);            % numAnchors × 1
            e = zNow - yPredWeighted;                                  % numAnchors × 1

            % -----------------------------------------------------------
            % 4. AdaBelief 모멘???�데?�트
            %      m(i) ??beta·m(i) + (1-beta)·e(i)
            %      s(i) ??beta·s(i) + (1-beta)·(e(i)-m(i))^2
            % -----------------------------------------------------------
            state.mMoment = obj.beta * state.mMoment + (1 - obj.beta) * e;
            state.sMoment = obj.beta * state.sMoment + (1 - obj.beta) * ((e - state.mMoment) .^ 2);

            % R inflation: R_k = R_nom + lambdaR * diag(s_k)
            state.diagR = state.nominalDiagR + obj.lambdaR * state.sMoment;

            % ?�치 ?�정?? ?��??�소 ?�리??
            state.diagR = min(max(state.diagR, obj.rFloor), obj.rCeil);

            % -----------------------------------------------------------
            % 5. ?�응??R�?가중치 ?�데?�트
            % -----------------------------------------------------------
            Rmat = diag(state.diagR);
            weightsUpd = obj.updateWeightsWithR(particlesPred, state.weights, zNow, Rmat);

            % -----------------------------------------------------------
            % 6. 추정 �?리샘?�링
            % -----------------------------------------------------------
            est = particlesPred * weightsUpd;
            [particlesRes, weightsRes, idxResampled, didResample] = obj.resampleEssWithIndices(particlesPred, weightsUpd);

            if didResample
                state.velPrev = particlesRes - state.particlesPrev(:, idxResampled);
            else
                state.velPrev = particlesRes - state.particlesPrev;
            end
            state.particlesPrev = particlesRes;
            state.weights       = weightsRes;
            state.estimatedPos(:, pointIdx) = est;
        end

        function weights = updateWeightsWithR(obj, particles, prevWeights, zNow, Rmat)
            % ?��??�서 R??받아 가?�시???�도�?가중치 ?�데?�트
            %   p(z|x_i) ??exp(-0.5 · eᵢ�? R?��?e�?
            yPred  = obj.H_nonlinear(particles);     % numAnchors × numParticles
            errors = zNow - yPred;                   % numAnchors × numParticles
            Rinv   = diag(1 ./ diag(Rmat));          % ?��??�렬?��?�???��?�을 ?�소�???���?

            distances = sum((Rinv * errors) .* errors, 1);  % 1 × numParticles

            weights = prevWeights(:)' .* exp(-0.5 * distances);
            weights = weights + 1e-300;
            weights = (weights / sum(weights)).';
        end
    end
end
