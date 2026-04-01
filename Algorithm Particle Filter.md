### Algorithm Particle Filter
$ m = 1, ... , N $
$$
\begin{array}{l} \\
\textbf{Particle filter }(X_{t-1}) \\
　X_t = 0 \\ \\

\textbf{1. Predict Particles(Sampling) } \\
　p{_m^-}=f(p_m) + \omega_{t-1}\\\\

\textbf{2. Update weights} \\
　w_m = w_m \cdot p(z_k|h(p_m^-)) \\
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


$ m = 1, ... , N $
$$
\begin{array}{l} \\
\textbf{Particle filter }(X_{t-1}) \\
　X_t = 0 \\ \\

\textbf{1. Predict Particles(Sampling) } \\
　p{_m^-}=f(p_m, \theta ) + \omega_{t-1}\\\\

\textbf{2. Update weights} \\
　w_m = w_m \cdot p(z_k|h(p_m^-)) \\
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


$x_t^{(i)}$를 대입합니다.<br>$\approx \mathcal{N}(x_t^{(i)}; \mu_t^{(i)}, \Sigma_t^{(i)})$

$x_t^{(i)}$를 대입합니다.<br>$\approx \mathcal{N}(x_t^{(i)}; f(x_{t-1}^{(i)}), Q)$