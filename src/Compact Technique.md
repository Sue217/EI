# Compact Technique

## Fundamental

> At the beginning, we have an $N\times D$ **PV** (Perturbation Vector), and we want to *compact* it to a $2\times D$ vector ($PV_C=[\mu_c, \sigma_c]$) to reduce the populations of the algorithm.

## Gauss Error Function

$$
erf(x)=\displaystyle\frac{2}{\sqrt{\pi}}\int^x_0e^{-t^2}dt
$$

<img src="/Users/sudo/Desktop/Research/src/figs/erf.png" alt="erf" style="zoom:67%;" />

## Formulas

$$
\displaystyle y = \sqrt2\delta\cdot erf^{-1}(x\cdot erf(\frac{\mu+1}{\sqrt2\delta})-x\cdot erf(\frac{\mu-1}{\sqrt2\delta})-erf(\frac{\mu+1}{\sqrt2\delta})+\mu\\\\
x_{actual}=y\times(ub-lb)/2+(ub+lb)/2\\\\
X_i(t),\ X_i(t+1) \larr GOA()\\\\
[winner, loser] = compare(x1,\ x2)\\\\
\mu_i^{c+1}=\mu_i^c+\frac{1}{N_p}(winner_i-loser_i)\\\\
\sigma_i^{c+1}=\sqrt{(\sigma_i^c)^2+(\mu_i^c)^2-(\mu_i^{c+1})^2+\frac{1}{N_p}(winner_i^2-loser_i^2)}
$$
