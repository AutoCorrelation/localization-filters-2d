# Problem 
- Circular-ToA based Localization  
- ToF를 측정하는데 송신, 수신 시간이 일치. (시간 동기화 or 같은 기기) 즉, Sonar, Rader
- 각 BS에 붙는 노이즈가 독립이라고 가정 가능. 

 #parameter  
 |x|number of x|
 |--|--|
 |Particles|5e3|
 |Iteration(KF)|1e3|
 |Iteration(PF)|1e3|  

 # results with 1e4 iteration  
=== RMSE Comparison ===
- Particle 150  

| Noise Level | Baseline | LinearKF | LinearPF | LinearKF_DecayQ | NonLinearPF |
|:-----------:|:--------:|:--------:|:--------:|:---------------:|:------------:|
| 1e-02  | 0.0997 | 0.0842 | 0.0864 | 0.0782 | 0.0780 |
| 1e-01  | 0.3193 | 0.2681 | 0.2754 | 0.2490 | 0.2473 |
| 1e+00  | 1.0076 | 0.8423 | 0.8694 | 0.7908 | 0.7763 |
| 1e+01  | 3.2520 | 2.5257 | 2.5922 | 2.4368 | 2.3032 |
| 1e+02  | 12.0150 | 6.9834 | 7.3520 | 6.5487 | 4.5284 |  

- Particle 1e3  

| Noise Level | Baseline | LinearKF | LinearPF | LinearKF_DecayQ | NonLinearPF |
|:-----------:|:--------:|:--------:|:--------:|:---------------:|:------------:|
| 1e-02   | 0.0997 | 0.0842 | 0.0851 | 0.0782     | 0.0763 |
| 1e-01   | 0.3193 | 0.2681 | 0.2714 | 0.2490     | 0.2422 |
| 1e+00   | 1.0076 | 0.8423 | 0.8512 | 0.7908     | 0.7586 |
| 1e+01   | 3.2520 | 2.5257 | 2.5393 | 2.4368     | 2.2371 |
| 1e+02   | 12.015 | 6.9834 | 6.8713 | 6.5487     | 4.0998 |
 
$$
Posterior = \frac{Likelihood(Observation) \times Prior(Transition)}{Proposal}
$$

$$
Bootstrap PF : q(x_k|x_{k-1},y_k)=p(x_k|x_{k-1})
$$

$$
w_m = w_m p(z_k|z_m^-)
$$