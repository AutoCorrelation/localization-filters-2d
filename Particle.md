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

### Algorithm Particle Filter
$ m = 1, ... , N $
$$
\begin{array}{l} \\
\textbf{Particle filter }(X_{t-1}) \\
　X_t = 0 \\ \\

\textbf{1. Predict Particles(Sampling) } \\
　p{_m^-}=f(p_m) \\\\

\textbf{2. Update weights} \\
　w_m = p(z_k|h(p_m^-)) \\
　w_m = {w_m}/{\Sigma w_m}  \\ \\

\textbf{3. Estimate state} \\
　\hat{x}_k = \Sigma^N_{m=1} p{_m^-}w_m \\ \\

\textbf{4. Resampling} \\
　\textbf{IF } ESS: 1/\Sigma w_m^2 < \frac {N}{2}\\
　　p_m = resample(p_m^-) \propto p(z_k|h(p_m^-))\\
　　w_m = 1/N \\
　\textbf{end} \\
\text{Add } \{p_m, w_m\} \text{ to } X_t\\ \\
\textbf{return }X_t \\

\end{array}
$$


---

### 3.1 Base로 보는 기준점
위 `Algorithm Particle Filter` 블록을 이 저장소에서는 `NonlinearParticleFilter`의 기준 pseudo code로 간주합니다.
아래는 같은 틀을 유지한 채, 각 구현에서 **달라진 부분만** 덧붙인 형태입니다.

---

### 3.2 LinearParticleFilter (측정 모델만 선형)
`NonlinearParticleFilter` 대비 변경점:

$$
\begin{array}{l}
	\textbf{[Step 2: Update weights 변경]} \\
	\text{Residual: } r_m = z_k - H\,p_m^- \\
	\text{Use data-driven } R_{k} = R\_{LLS}(:,:,pointIdx,iterIdx) \\
w_m \propto w_m \cdot \exp\!\left(-\frac{1}{2} r_m^\top R_k^{-1} r_m\right)
\end{array}
$$

핵심: 비선형 거리 측정 `h(\cdot)` 대신 선형 관측식 `H x`와 시점별 `R_LLS`를 사용합니다.

---

### 3.3 CustomNonlinearParticleFilter (현재 코드 기준 최소 변경)
`NonlinearParticleFilter` 대비 변경점:

$$
\begin{array}{l}
	\textbf{[Init 추가]} \\
	\text{state.M} \leftarrow \mathbf{0}_{4 \times N},\quad \text{state.S} \leftarrow \mathbf{0}_{4 \times N} \\
	\textbf{[Update/Resampling]} \\
	\text{기본 NonlinearParticleFilter와 동일}
\end{array}
$$

핵심: 초기 모멘트 상태만 추가되고, `step`은 부모 로직을 그대로 사용합니다.

---

### 3.4 EKFParticleFilter (EKF proposal 분포)
`NonlinearParticleFilter` 대비 변경점:

$$
\begin{array}{l}
	\textbf{[Step 1 변경: Predict/Sampling]} \\
\mu_i^- = p_{i,k-1} + v_{i,k-1} + b \\
q(x_k|x_{k-1},z_k)=\mathcal{N}(\mu_{q,i}, P_{q,i}) \text{ 를 EKF로 구성} \\
p_{i,k}^- \sim \mathcal{N}(\mu_{q,i}, P_{q,i}) \\
	\textbf{[Step 2 변경: Weight]} \\
\log w_i \leftarrow \log w_i + \log p(z_k|p_{i,k}^-) + \log p(p_{i,k}^-|\mu_i^-) - \log q(p_{i,k}^-|\mu_i^-,z_k)
\end{array}
$$

핵심: 단순 process 샘플링 대신 EKF 기반 proposal을 만들고, importance ratio에 `prior/proposal` 보정항이 들어갑니다.

---

### 3.5 AdaptiveParticleFilter (AdaBelief 기반 R 적응)
`NonlinearParticleFilter` 대비 변경점:

$$
\begin{array}{l}
	\textbf{[Init 추가]} \\
	\text{diag}R_{nom},\; m,\; s \text{ 초기화} \\
	\textbf{[Step 2 이전 추가: 전역 잔차 모멘트]} \\
\hat{x}_k = \sum_i w_i p_{i,k}^- ,\quad \hat{y}_k = h(\hat{x}_k),\quad e_k = z_k-\hat{y}_k \\
m \leftarrow \beta m + (1-\beta)e_k \\
s \leftarrow \beta s + (1-\beta)(e_k-m)^2 \\
	\text{diag}R_k \leftarrow \text{clip}(\text{diag}R_{nom}+\lambda_R s,\; r_{floor},\; r_{ceil}) \\
	\textbf{[Step 2 변경]} \\
w_m \propto w_m \cdot \exp\!\left(-\frac{1}{2}(z_k-h(p_m^-))^\top R_k^{-1}(z_k-h(p_m^-))\right)
\end{array}
$$

핵심: 측정 공분산 `R`을 고정하지 않고 잔차 통계로 매 스텝 갱신합니다.

---

### 3.6 BeliefQShrinkAdaptiveParticleFilter (belief에 따른 Q 축소)
`AdaptiveParticleFilter` 대비 변경점:

$$
\begin{array}{l}
	\textbf{[Step 1 변경]} \\
\rho = \mathrm{mean}\!\left(\frac{s}{\max(\text{diag}R_{nom}, r_{floor})}\right) \\
q_{var} = \max\!\left(\frac{1}{1+g_q\rho},\; q_{min}^2\right),\quad q_{scale}=\sqrt{q_{var}} \\
p_m^- = p_m + v_m + b + q_{scale}\,\epsilon_m
\end{array}
$$

핵심: belief ratio가 커질수록 process 확산을 줄이는 방향으로 `Q`를 스케일링합니다.

---

### 3.7 BeliefRougheningAdaptiveParticleFilter (belief에 따른 roughening 강도 조절)
`AdaptiveParticleFilter` 대비 변경점:

$$
\begin{array}{l}
	\textbf{[Step 4 이후 추가]} \\
\rho = \mathrm{mean}\!\left(\frac{s}{\max(\text{diag}R_{nom}, r_{floor})}\right) \\
k_{rough} = \min\left(k_0(1+g_r\rho),\; k_{max}\right) \\
	\textbf{if resampled: } p_m \leftarrow p_m + \sigma_{rough}\odot \eta_m \\
\sigma_{rough} = k_{rough}\,\text{span}(P)\,N^{-1/n}
\end{array}
$$

핵심: resampling 뒤에 주입하는 roughening 노이즈 세기를 belief 상태로 동적으로 조절합니다.

---

### 3.8 RDiagPriorEditAdaptiveParticleFilter (Adaptive R + Prior Editing)
`AdaptiveParticleFilter` 대비 변경점:

$$
\begin{array}{l}
	\textbf{[Step 2 직후 추가: Prior Editing]} \\
\sigma_{gate,j} = g_{\sigma}\sqrt{\max((\text{diag}R_k)_j, r_{floor})} \\
	\text{Reject particle if } |z_{k,j}-h_j(p_m^-)| > \sigma_{gate,j} \; (\exists j) \\
	\text{For rejected } m:\; \text{repropagate up to } T_{max} \text{ and keep best candidate} \\
	\textbf{[Step 2]} \text{는 편집된 } p_m^- \text{에 대해 adaptive } R_k \text{로 수행}
\end{array}
$$

핵심: adaptive `R_k`로 게이트를 축별로 정의하고, 벗어난 파티클만 재전파하여 사전 분포를 편집합니다.

---

### 3.9 RougheningPriorEditingParticleFilter (고정 R 계열의 Prior Editing + Roughening)
`NonlinearParticleFilter` 대비 변경점:

$$
\begin{array}{l}
	\textbf{[Step 2 이전 추가: Prior Editing]} \\
	\text{Gate} = g_{\sigma}\cdot \sigma_{noise} \\
	\text{Reject particle if } \max_j |z_{k,j}-h_j(p_m^-)| > \text{Gate} \\
	\text{Rejected particle 재전파(최대 } T_{max}\text{)} \\
	\textbf{[Step 4 이후 추가: Roughening]} \\
p_m \leftarrow p_m + \sigma_{rough}\odot \eta_m,
\quad \sigma_{rough}=k\,\text{span}(P)\,N^{-1/n}
\end{array}
$$

핵심: adaptive `R` 없이, 고정 노이즈 스케일 기반 게이팅 + roughening을 결합한 휴리스틱 보강형 PF입니다.

## 4. 고급 적응형 기법 (Adaptive PF)
제공된 코드에서는 성능을 극대화하기 위해 다음과 같은 진보된 기법들이 구현되어 있습니다.

### Adaptive R (측정 노이즈 적응)
글로벌 잔차(Residual)의 2차 모멘트(EMA)를 실시간으로 추적하여 측정 노이즈 공분산 $R$을 동적으로 조절합니다.
* **물리적 효과**: 센서 노이즈가 갑자기 튀는 상황(Outlier)에서 $R$을 팽창시켜 필터의 발산을 방지합니다.

### AdaBelief 기반 모멘트 활용
`CustomNonlinearParticleFilter.m`에서는 최적화 알고리즘인 AdaBelief의 개념을 도입하여 잔차의 1차($M$) 및 2차($S$) 모멘트를 관리합니다.
* **방향성 보정**: 1차 모멘트($m_k$)를 통해 파티클이 이동해야 할 경향성을 파악합니다.

---