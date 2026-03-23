파티클 필터(Particle Filter, PF)는 시스템의 상태를 확률 분포로 추정하는 순차적 몬테카를로(Sequential Monte Carlo, SMC) 방법의 일종입니다. 특히 시스템이 비선형적이거나 노이즈가 가우시안 분포를 따르지 않을 때 강력한 성능을 발휘합니다.

---

## 1. 파티클 필터의 핵심 철학
파티클 필터는 "수많은 점(Particles)을 뿌려보고, 실제 관측값과 비슷한 위치에 있는 점들에게 더 높은 점수를 주자"는 직관에서 시작합니다.

* **샘플 기반 표현**: 상태 확률 밀도 함수를 고정된 수식 대신 무작위로 생성된 수많은 파티클들의 집합으로 표현합니다.
* **비선형성 대응**: 칼만 필터와 달리 상태 전이 모델이나 관측 모델이 비선형적이어도 선형화(Linearization) 과정 없이 직접 적용 가능합니다.

---

## 2. 수학적 프레임워크 (Bayesian Filtering)
파티클 필터는 베이즈 정리(Bayes' Rule)를 재귀적으로 적용하여 사후 확률(Posterior)을 추정합니다.

$$Posterior = \frac{Likelihood(Observation) \times Prior(Transition)}{Proposal}$$

주요 수식 및 변수의 물리적 의미는 다음과 같습니다:
* $p(x_k | x_{k-1})$: **상태 전이 모델 (Prior)**. 이전 상태에서 현재 상태로 어떻게 움직일지 예측합니다.
* $p(z_k | x_k)$: **우도 함수 (Likelihood)**. 특정 파티클 위치에서 현재 관측값($z_k$)이 나올 확률을 계산합니다.
* $w_k^{(i)}$: **중요도 가중치 (Importance Weight)**. 관측값과 유사할수록 해당 파티클에 부여되는 가중치가 커집니다.

---

## 3. 필터링 프로세스 (Step-by-Step)

### 단계 1: 예측 (Prediction)
이전 단계의 파티클들을 시스템 모델에 따라 이동시키고 프로세스 노이즈를 추가합니다.
* `particlesPred = state.particlesPrev + state.velPrev + obj.sampleProcess();`

### 단계 2: 가중치 업데이트 (Update)
각 파티클이 실제 측정값과 얼마나 일치하는지 비교하여 가중치를 부여합니다.
* **선형 업데이트**: $z = Hx$ 관계를 이용한 가우시안 우도 계산.
* **비선형 업데이트**: 거리 함수($\sqrt{dx^2 + dy^2}$) 등을 직접 적용하여 오차 계산.

### 단계 3: 리샘플링 (Resampling)
가중치가 낮은 파티클은 제거하고, 가중치가 높은 파티클 위주로 다시 복제하여 샘플 빈약화(Degeneracy) 문제를 해결합니다.
* **ESS (Effective Sample Size)**: 파티클들의 유효성을 평가하여 임계값보다 낮을 때만 리샘플링을 수행합니다.

---

## 4. 고급 적응형 기법 (Adaptive PF)
제공된 코드에서는 성능을 극대화하기 위해 다음과 같은 진보된 기법들이 구현되어 있습니다.

### Adaptive R (측정 노이즈 적응)
글로벌 잔차(Residual)의 2차 모멘트(EMA)를 실시간으로 추적하여 측정 노이즈 공분산 $R$을 동적으로 조절합니다.
* **물리적 효과**: 센서 노이즈가 갑자기 튀는 상황(Outlier)에서 $R$을 팽창시켜 필터의 발산을 방지합니다.

### AdaBelief 기반 모멘트 활용
`CustomNonlinearParticleFilter.m`에서는 최적화 알고리즘인 AdaBelief의 개념을 도입하여 잔차의 1차($M$) 및 2차($S$) 모멘트를 관리합니다.
* **방향성 보정**: 1차 모멘트($m_k$)를 통해 파티클이 이동해야 할 경향성을 파악합니다.

---

daptiveParticleFilter, BeliefQShrinkAdaptiveParticleFilter, BeliefRougheningAdaptiveParticleFilter, RDiagPriorEditAdaptiveParticleFilter, ResidualSquaredAdaptiveParticleFilter