# Filters (src/Filters) — Idea Summary

이 문서는 `src/Filters/` 폴더에 구현된 "내 아이디어 기반 필터들"의 핵심 개념을 한 장으로 정리한 것입니다. 
목표는 **(1) 어떤 가정(동역학/관측)을 두는지**, **(2) 무엇을 바꿔서(적응/강인화/다양성 유지) 성능을 올리는지**, **(3) 비용-성능 트레이드오프가 무엇인지**를 빠르게 비교하고, 다음 실험/새 아이디어 도출의 발판을 만드는 것입니다.

---

## 0) 공통 전제(코드에서 실제로 공유하는 구조)

### 상태 및 동역학(대부분 공통)

- **위치 상태**: 2D `x = [x; y]`
- **속도 모델**: 별도 상태로 엄밀히 모델링하기보다는, **직전 추정/입자 위치 차분으로 유지**
- **예측 단계**:

$$\textbf{Prediction:}\quad x_k = x_{k-1} + v_{k-1} + b + \epsilon_k$$

  - `b`: `processBias` (데이터에서 읽어옴)
  - `ε_k`: 과정잡음(데이터의 noise bank 또는 Gaussian)

### 리샘플링(Particle Filter 계열 공통)

- **ESS 기준 리샘플링**:
  - $\text{ESS} = 1/\sum_i w_i^2$
  - $\text{ESS} < N \times \text{resampleThresholdRatio}$이면 리샘플
- **리샘플 후**: 가중치는 균등 $(1/N)$
- **속도 갱신**: "리샘플 전-후 입자 이동량"으로 업데이트

### 관측 모델의 2가지 축

| 축 | 설명 | 사용 변수 |
|---|---|---|
| **Linear (LLS 기반)** | 선형 측정 모델 | `z_LLS`, `H`, `R_LLS` |
| **Nonlinear (앵커-거리 ranging)** | 거리 기반 관측 | `ranging`, 앵커 좌표 `Anchor` |

---

## 1) 필터별 개념 요약

### 1. Baseline — `Baseline.m`

| 항목 | 설명 |
|---|---|
| **관측** | `z_LLS`를 `pinv(H)`로 즉시 위치로 투영 |
| **특징** | 동역학/잡음 적응/리샘플링 없음 (가장 단순한 기준선) |
| **장점** | 매우 빠름 |
| **단점** | 시간적 일관성(동역학) 활용 없음, 이상치/노이즈 변화에 취약 |

---

### 2. LinearKalmanFilter — `LinearKalmanFilter.m`

| 항목 | 설명 |
|---|---|
| **모델** | 선형 Kalman Filter |
| **예측** | `xPred = xPrev + vPrev + processBias`, `PPred = PPrev + Q` |
| **업데이트** | `K = PPred H' (H PPred H' + R)^(-1)` |
| **관측** | `z_LLS`, `H`, `R_LLS` |
| **의도** | 선형 가정 하에서 최소분산 추정 |

---

### 3. LinearKalmanFilter_DecayQ — `LinearKalmanFilter_DecayQ.m`

| 항목 | 설명 |
|---|---|
| **아이디어** | 시간이 갈수록 과정잡음 `Q`를 감소시켜 후반을 더 안정화 |
| **공식** | `PPred = PPrev + Q * exp(-gamma * (pointIdx - 3))` |
| **장점** | 후반 드리프트/진동 감소 가능 |
| **단점** | 급격한 동역학 변화(실제 움직임 변화)에 둔감해질 수 있음 |

---

### 4. LinearParticleFilter — `LinearParticleFilter.m`

| 항목 | 설명 |
|---|---|
| **유형** | Particle Filter 기본형 (선형 관측) |
| **예측** | `xPred = xPrev + vPrev + bias + processNoise` |
| **가중치** | `w_i ∝ w_i * exp(-0.5 * (z - Hx_i)^T R^(-1) (z - Hx_i))` |
| **리샘플링** | ESS 기반 |
| **특이점** | noise bank 지원 (`processNoise`, `toaNoise`)으로 데이터 기반 샘플링 가능 |

---

### 5. NonlinearParticleFilter — `NonlinearParticleFilter.m`

| 항목 | 설명 |
|---|---|
| **유형** | Particle Filter 기본형 (비선형 관측) |
| **관측 모델** | 각 앵커까지 거리: $h(x) = [\|x-a_1\|, \|x-a_2\|, \dots]^T$ |
| **가중치** | `zNow - h(x_i)`의 제곱합 기반 likelihood |
| **의도** | 선형 LLS 관측이 아니라 ranging 관측을 직접 사용 |

---

### 6. EKFParticleFilter — `EKFParticleFilter.m`

| 항목 | 설명 |
|---|---|
| **핵심 아이디어** | Proposal을 prior가 아니라 **EKF로 만든 posterior 근사**에서 샘플링 |
| **Proposal 분포** | $q(x_k \mid x_{k-1}, z_k) = \mathcal{N}(\mu_q, P_q)$ |
| **가중치 보정** | $w \propto w_{\text{prev}} \cdot p(z\mid x) \cdot p(x\mid x_{\text{prev}}) / q(x\mid x_{\text{prev}}, z)$ |
| **장점** | 측정 정보가 강할수록 샘플 효율↑ (필요 입자수↓ 가능) |
| **단점** | 입자당 EKF 수행 → 연산량↑, 수치 불안정 방어 필요(jitter/robust Cholesky) |

---

### 7. AdaptiveParticleFilter — `AdaptiveParticleFilter.m`

| 항목 | 설명 |
|---|---|
| **핵심 아이디어** | Innovation(잔차)의 AdaBelief 모멘트로 **측정 공분산 R을 적응적으로 팽창** |
| **잔차** | $e_k = z_k - h(\hat{x}_k)$, where $\hat{x}_k = \sum_i w_i x_i$ |
| **1차 모멘트** | $m_k \leftarrow \beta m_{k-1} + (1-\beta)e_k$ |
| **2차 모멘트** | $s_k \leftarrow \beta s_{k-1} + (1-\beta)(e_k - m_k)^2$ |
| **R 적응** | $R_k = R_0 + \lambda_R \,\text{diag}(s_k)$ |
| **의도** | 고정 R이 실제 오차 분포/모델 미스매치를 못 따라갈 때, likelihood를 자동 완화해 붕괴 방지 |
| **주의** | "입자분포 전체"가 아니라 "가중평균 1점"에서 잔차를 만들기 때문에, 멀티모달/비대칭 분포에서는 과/미적응 가능 |

---

### 8. BeliefQShrinkAdaptiveParticleFilter — `BeliefQShrinkAdaptiveParticleFilter.m`

| 항목 | 설명 |
|---|---|
| **핵심 아이디어** | Belief(= sMoment 크기)가 크면 **과정잡음을 축소(Q shrink)** |
| **직관** | 측정이 불확실한 상황에서 상태 예측까지 크게 퍼뜨리면 추정이 더 흔들릴 수 있으니 안정화 |
| **효과 기대** | Drift/난조 구간에서 PF의 과도한 확산을 줄여 APE 개선 가능 |

---

### 9. RDiagPriorEditAdaptiveParticleFilter — `RDiagPriorEditAdaptiveParticleFilter.m`

| 항목 | 설명 |
|---|---|
| **핵심 아이디어** | 적응적으로 커진 `diag(R_k)`를 이용해 **앵커별 게이팅(sigma gate)**을 만들고, gate 밖 입자는 "재시도 샘플링"으로 **prior editing** |
| **로직** | 1. Gate 밖 입자 검출 → 2. 새 샘플링 시도 → 3. 실패 시 `priorMaxRetry` 반복 → 4. 여전히 실패하면 원본 유지 (낮은 가중치) |
| **강점** | Weight로만 벌주기 전에 "말이 되는 후보로 교체" → 가중치 붕괴/이상치에 강함 |
| **비용** | Reject된 입자마다 최대 `priorMaxRetry` 반복 → worst-case 런타임 증가 |

---

### 10. BeliefRougheningAdaptiveParticleFilter — `BeliefRougheningAdaptiveParticleFilter.m`

| 항목 | 설명 |
|---|---|
| **핵심 아이디어** | BeliefRatio가 크면 roughening 강도(K)를 키워 리샘플 이후 다양성 회복 |
| **적용 조건** | 리샘플이 발생했을 때만 적용 |
| **효과 기대** | Sample impoverishment 완화 |
| **리스크** | 불필요한 roughening은 분산만 키워 APE 악화 가능 → belief 기반으로 "필요할 때만"이 포인트 |

---

### 11. CustomNonlinearParticleFilter — `CustomNonlinearParticleFilter.m`

| 항목 | 설명 |
|---|---|
| **현재 상태** | NonlinearPF 기반의 실험용 스캐폴딩 |
| **상태** | AdaBelief 모멘트 관련 함수/상태 변수가 있으나, `step()`에 통합된 최종 알고리즘 형태는 아직 미완성에 가까움 |

---

## 2) 현재 아이디어들을 관통하는 설계 축(axes)

### 축 1: 관측 강인성 (Robustness to Observation Mismatches)

$$\text{고정 likelihood (NonlinearPF)} \to \text{R 적응 (AdaptivePF)} \to \text{게이팅+교정 (RDiagPriorEdit)}$$

### 축 2: 샘플 효율 (Importance Proposal Quality)

$$\text{Prior proposal (기본 PF)} \to \text{EKF proposal (EKFParticleFilter)}$$

### 축 3: 다양성 유지 (Particle Diversity)

$$\text{ESS 리샘플만} \to \text{Belief 기반 roughening}$$

### 축 4: 동역학-관측 결합 (Complementary Control)

- R을 키울 때 Q를 줄이기 (BeliefQShrink)
- 반대로 "필요 시 Q를 키워 재탐색" 같은 쌍으로 설계 가능

---

## 3) 새 아이디어 도출(현재 코드 철학을 자연스럽게 확장)

### A. Belief 계산을 "1점(가중평균)"에서 "분포(입자)"로 확장

**문제**
- AdaptivePF는 `e_k = z - h(Σ w x)` 하나로 belief를 업데이트
- 멀티모달일 때 잘못된 신호 가능

**후보 방안**
- 입자별 residual의 **가중 중앙값/가중 MAD**로 anchor별 스케일을 추정
- `Var[h(x_i)]`(예측 관측의 분산) 기반으로 R inflation 또는 게이팅을 조절

---

### B. EKF Proposal의 조건부 사용(혼합 Proposal)

- 항상 EKF proposal은 비싸므로,
  - `ESS`가 낮거나,
  - `beliefRatio`가 높을 때만 EKF proposal을 켜는 스위치/혼합을 고려

---

### C. 앵커별 신뢰도 학습/완화(Soft Anchor Dropout)

- `diagR`가 지속적으로 큰 anchor가 있으면 해당 anchor의 likelihood 기여를 완화하거나, 일정 기간 다운웨이트

---

### D. Likelihood 자체를 강인화(Student-t / Huber Loss)

- R inflation은 "Gaussian 폭"을 늘리는 방식
- 다른 접근으로 잔차가 큰 경우 페널티를 선형/완만하게 만드는 heavy-tail likelihood를 적용 가능

---

### E. Targeted Roughening / Targeted Prior Editing

- 전체 입자에 roughening이 아니라, gate 밖 또는 low-likelihood subset에만 선택적으로 적용해 불필요 확산을 줄임

---

## 4) 실험 설계(비교할 때 최소 체크리스트)

- ✓ **APE vs Runtime** 동시 기록 (이미 파이프라인 지원)
- ✓ **Noise level별 튜닝** 필요한 변형이면 `getBestParams` 같은 helper로 분리
- ✓ **핵심 하이퍼파라미터** 적응형 계열:
  - `beta` (모멘트 감마)
  - `lambdaR` (R inflation 강도)
  - `priorSigmaGate` (게이팅 문턱)
  - `K` (roughening 강도)

---

## 5) 관련 기본 설정 위치

- **기본 config 값**: `src/utils/initializeConfig.m`
  - `resampleThresholdRatio`
  - `decayGamma`
  - Belief/adaptive 관련 기본 파라미터
  - EKF-proposal 옵션 등

