classdef AdaptiveRParticleFilter < NonlinearParticleFilter
    % AdaptiveRParticleFilter  잔차의 2차 모멘트(EMA)를 이용한 적응형 R 파티클 필터
    %
    % [수학적 배경]
    %   표준 PF에서는 고정된 R = sigma^2 * I 를 우도 함수에 사용한다.
    %   R을 인위적으로 팽창(inflation)하면 샘플 빈약화(sample impoverishment)에
    %   강인해지지만, 점근적 최적성(asymptotic optimality)을 상실한다.
    %
    %   이 필터는 매 스텝에서 잔차의 2차 모멘트를 추적해 R을 실시간 스케일링한다.
    %
    %     e_k   = z_k - H(x̂_k)                      (가중 평균 예측 잔차)
    %     r̂_k(i) = α·r̂_{k-1}(i) + (1-α)·e_k(i)²    (앵커 i별 분산 EMA 업데이트)
    %     R_k   = diag(r̂_k(1), ..., r̂_k(n))          (대각 R 재구성)
    %
    %   이후 가중치 업데이트는 R_k를 이용한 가우시안 우도로 수행한다.
    %     w_i ∝ w_{i-1} · exp(-0.5 · eᵢᵀ R_k⁻¹ eᵢ)
    %
    % [참고] alpha가 1에 가까울수록 과거 추정치를 강하게 유지(느린 적응),
    %        0에 가까울수록 현재 잔차에 민감하게 반응(빠른 적응).
    %
    % [사용 예]
    %   filterObj = AdaptiveRParticleFilter(data, config, noiseIdx);        % alpha=0.9
    %   filterObj = AdaptiveRParticleFilter(data, config, noiseIdx, 0.8);  % alpha=0.8

    properties
        % EMA 망각 인자 (forgetting factor): 0 < alpha < 1
        alpha   (1,1) double = 0.9

        % R 대각 요소의 하한 / 상한 (수치 안정성 보장)
        rFloor  (1,1) double = 1e-6
        rCeil   (1,1) double = 1e4
    end

    methods
        function obj = AdaptiveRParticleFilter(data, config, noiseIdx, alpha)
            % 생성자
            %   alpha (선택): EMA 망각 인자, 기본값 0.9
            obj@NonlinearParticleFilter(data, config, noiseIdx);

            if nargin >= 4 && ~isempty(alpha)
                obj.alpha = alpha;
            end
        end

        function state = initializeState(obj, numPoints)
            % 부모 클래스 state + 적응형 R 상태 추가
            state = initializeState@NonlinearParticleFilter(obj, numPoints);

            % 초기 diagR: 명목 noiseVariance 기반
            numAnchors = size(obj.anchorPos, 1);
            nominalVar = obj.noiseScale^2;                    % noiseVariance(noiseIdx)
            state.diagR = nominalVar * ones(numAnchors, 1);   % 앵커별 분산 (numAnchors × 1)
        end

        function [state, est] = step(obj, state, iterIdx, pointIdx)
            % -----------------------------------------------------------
            % 1. 상태 예측 (부모와 동일)
            % -----------------------------------------------------------
            particlesPred = state.particlesPrev + state.velPrev + obj.sampleProcess();

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
            % 4. 잔차 2차 모멘트(분산) EMA 업데이트
            %      r̂(i) ← α·r̂(i) + (1-α)·e(i)²
            % -----------------------------------------------------------
            state.diagR = obj.alpha * state.diagR + (1 - obj.alpha) * (e .^ 2);

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
