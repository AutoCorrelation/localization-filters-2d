classdef AdaptiveParticleFilter < NonlinearParticleFilter
    % AdaptiveParticleFilter  AdaBelief 스타일 R-inflation 파티클 필터
    %
    % [수학적 배경]
    %   표준 PF에서는 고정된 R = sigma^2 * I 를 우도 함수에 사용한다.
    %   R을 인위적으로 팽창(inflation)하면 샘플 빈약화(sample impoverishment)에
    %   강인해지지만, 점근적 최적성(asymptotic optimality)을 상실한다.
    %
    %   이 필터는 매 스텝에서 글로벌 잔차의 2차 모멘트 s_k를 추적하고,
    %   R_k = R_nom + lambdaR * diag(s_k) 로 관측 공분산을 팽창시킨다.
    %
    %     e_k    = z_k - H(x̂_k)                            (가중 평균 예측 잔차)
    %     s_k(i) = beta·s_{k-1}(i) + (1-beta)·e_k(i)^2     (2차 모멘트 EMA)
    %     R_k    = R_nom + lambdaR * diag(s_k)             (대각 팽창)
    %
    %   이후 가중치 업데이트는 R_k를 이용한 가우시안 우도로 수행한다.
    %     w_i ∝ w_{i-1} · exp(-0.5 · eᵢᵀ R_k⁻¹ eᵢ)
    %
    % [참고] beta가 1에 가까울수록 과거 추정치를 강하게 유지(느린 적응),
    %        0에 가까울수록 현재 잔차에 민감하게 반응(빠른 적응).
    %
    % [사용 예]
    %   filterObj = AdaptiveParticleFilter(data, config, noiseIdx);             % beta=0.99, lambdaR=1.0
    %   filterObj = AdaptiveParticleFilter(data, config, noiseIdx, 0.8, 2.0);  % beta=0.8, lambdaR=2.0

    properties
        % AdaBelief 계수(EMA 망각 인자): 0 < beta < 1
        beta    (1,1) double = 0.99

        % R inflation 강도: R_k = R_nom + lambdaR * diag(s_k)
        lambdaR (1,1) double = 1.0

        % R 대각 요소의 하한 / 상한 (수치 안정성 보장)
        rFloor  (1,1) double = 1e-6
        rCeil   (1,1) double = 1e4
    end

    methods
        function obj = AdaptiveParticleFilter(data, config, noiseIdx, beta, lambdaR)
            % 생성자
            %   beta (선택): AdaBelief 계수, 기본값 0.99
            %   lambdaR (선택): R inflation 강도, 기본값 1.0
            obj@NonlinearParticleFilter(data, config, noiseIdx);

            if nargin >= 4 && ~isempty(beta)
                obj.beta = beta;
            end
            if nargin >= 5 && ~isempty(lambdaR)
                obj.lambdaR = lambdaR;
            end
        end

        function state = initializeState(obj, numPoints)
            % 부모 클래스 state + 적응형 R 상태 추가
            state = initializeState@NonlinearParticleFilter(obj, numPoints);

            % 명목 R 대각 및 2차 모멘트 상태 초기화
            numAnchors = size(obj.anchorPos, 1);
            nominalVar = obj.noiseScale^2;                    % noiseVariance(noiseIdx)
            state.nominalDiagR = nominalVar * ones(numAnchors, 1);
            state.mMoment = zeros(numAnchors, 1);
            state.sMoment = zeros(numAnchors, 1);
            state.diagR = state.nominalDiagR;
        end

        function [state, est] = step(obj, state, iterIdx, pointIdx)
            % -----------------------------------------------------------
            % 1. 상태 예측 (부모와 동일)
            % -----------------------------------------------------------
            particlesPred = state.particlesPrev + state.velPrev + obj.processBias + obj.sampleProcess();

            % -----------------------------------------------------------
            % 2. 측정값
            % -----------------------------------------------------------
            zNow = obj.z(:, pointIdx, iterIdx);

            % -----------------------------------------------------------
            % 3. 가중 평균 파티클로 잔차 계산
            %      x̂_k = Σ w_i · x_i
            %      ŷ_k = H(x̂_k)  →  e_k = z_k - ŷ_k
            % -----------------------------------------------------------
            xHatWeighted  = particlesPred * state.weights;            % 2 × 1
            yPredWeighted = obj.H_nonlinear(xHatWeighted);            % numAnchors × 1
            e = zNow - yPredWeighted;                                  % numAnchors × 1

            % -----------------------------------------------------------
            % 4. AdaBelief 2차 모멘트 업데이트
            %      s(i) ← beta·s(i) + (1-beta)·e(i)^2
            % -----------------------------------------------------------
            state.mMoment = obj.beta * state.mMoment + (1 - obj.beta) * e;
            state.sMoment = obj.beta * state.sMoment + (1 - obj.beta) * ((state.mMoment- e ).^ 2);

            % R inflation: R_k = R_nom + lambdaR * diag(s_k)
            state.diagR = state.nominalDiagR + obj.lambdaR * state.sMoment;

            % 수치 안정성: 대각 요소 클리핑
            state.diagR = min(max(state.diagR, obj.rFloor), obj.rCeil);

            % -----------------------------------------------------------
            % 5. 적응된 R로 가중치 업데이트
            % -----------------------------------------------------------
            Rmat = diag(state.diagR);
            weightsUpd = obj.updateWeightsWithR(particlesPred, state.weights, zNow, Rmat);

            % -----------------------------------------------------------
            % 6. 추정 및 리샘플링
            % -----------------------------------------------------------
            est = particlesPred * weightsUpd;
            [particlesRes, weightsRes] = obj.resampleEss(particlesPred, weightsUpd);

            state.velPrev       = est * ones(1, obj.numParticles) - state.particlesPrev;
            state.particlesPrev = particlesRes;
            state.weights       = weightsRes;
            state.estimatedPos(:, pointIdx) = est;
        end

        function weights = updateWeightsWithR(obj, particles, prevWeights, zNow, Rmat)
            % 외부에서 R을 받아 가우시안 우도로 가중치 업데이트
            %   p(z|x_i) ∝ exp(-0.5 · eᵢᵀ R⁻¹ eᵢ)
            yPred  = obj.H_nonlinear(particles);     % numAnchors × numParticles
            errors = zNow - yPred;                   % numAnchors × numParticles
            Rinv   = diag(1 ./ diag(Rmat));          % 대각 행렬이므로 역행렬을 원소별 역수로

            distances = sum((Rinv * errors) .* errors, 1);  % 1 × numParticles

            weights = prevWeights(:)' .* exp(-0.5 * distances);
            weights = weights + 1e-300;
            weights = (weights / sum(weights)).';
        end
    end
end
