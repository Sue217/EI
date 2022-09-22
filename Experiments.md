## Experiments

### Pseudo-code of PGOA

```pseudocode
Require: Parameters: N, D, lb, ub, G, St, Mg, Cp, Co.
Ensure: Global optimum global_best and its fitness value global_min.
Set the group number G, then each group presents G_i(i ≤ G).
Initialize G[i], G[i].best, G[i].min, G[i].pop_fit of each group.
Initialize Np = N / G, T = max_iter / Co, n = log(G), m = 0.

for iter in 1 to max_iter:
  for g in G:
    if rand > 0.5:
      do Exploration
    else:
      do Exploitation
    for i in Np:
      Update G[g].pop_fit[i], G[g].best, G[g].min, global_best, global_min
    if communication:
      while select groups randomly:
        q = g ^ (2**m)
        sorted_pop_fit = reverse(sort(pop_fit[q]))
        expected_pop_fit = sorted_pop_fit[Np * migration]
        Update(where fitness_func(X[q]) >= expected_pop_fit) = group_best[g]
```

### Uni-modal Test Functions



|  ID  | Function                                                     |  LB   |  UB  | Dim  |
| :--: | :----------------------------------------------------------- | :---: | :--: | :--: |
|  F1  | $f(x) = \displaystyle \sum_{i=1}^D x_i^2$                    | -100  | 100  |  50  |
|  F2  | $f(x) = \displaystyle \sum_{i=1}^D\ \lvert x_i \lvert + \prod_{i=1}^D\  \lvert x_i \lvert$ |   5   |  10  |  50  |
|  F3  | $f(x) = \displaystyle \sum_{i=1}^D (\sum_{j=1}^i x_j)$       | -100  | 100  |  30  |
|  F4  | $f(x) = \max \{\lvert x_i \lvert ,\ 1 \leq i \leq n \}$      | -100  | 100  |  50  |
|  F5  | $f(x) = \displaystyle \sum_{i=1}^{D-1}\ [100(x_{i+1}-x_i^2)^2+(x_i-1)^2]$ |  -5   |  10  |  50  |
|  F6  | $f(x) = \displaystyle \sum_{i=1}^D (x_i + 0.5)^2$            | -100  | 100  |  50  |
|  F7  | $f(x) = \displaystyle \sum_{i=0}^D\ i \cdot x_i^4 + rand(0,1)$ | -1.28 | 1.28 |  30  |



### Multi-modal Test Functions



| ID   | Function                                                     | LB   | UB   | Dim  |
| :--- | :----------------------------------------------------------- | ---- | ---- | ---- |
| F8   | $f(x) = \displaystyle \sum_{i=1}^D (-x \cdot \sin(\sqrt{\lvert x_i \lvert}))$ | -500 | 500  | 50   |
| F9   | $f(x) = 10D + \displaystyle \sum_{i=1}^D [x_i^2-10 \cos(2 \pi x_i)]$ | 2.56 | 5.12 | 50   |
| F10  | $f(x) = \displaystyle -20 \exp(-0.2 \sqrt{\frac{1}{n} \sum_{i=1}^D x_i^2}) - \exp(\frac{1}{n} \sum_{i=1}^D \cos(2 \pi x_i)) + 20 + e$ | -32  | 32   | 50   |
| F11  | $f(x) = \displaystyle \frac{1}{4000} \sum_{i=1}^D x_i^2 - \prod_{i=1}^D \cos (\frac{x_i}{\sqrt{i}}) + 1$ | 300  | 600  | 50   |
| F12  | $f(x) = \displaystyle \frac{\pi}{D} \{10 \cdot \sin(\pi y_1) \} + \sum_{i=1}^{D-1}(y_i - 1)^2[1 + 10 \sin^2(\pi y_i + 1) + \sum_{i=1}^D u(x_i, 10, 100, 4)],\ where\ y_i = 1 + \frac{x_i+1}{4}$ | -50  | 50   | 30   |
| F13  | $f(x) = \displaystyle 0.1(\sin^2 (3 \pi x_1) + \sum_{i=1}^D(x_i-1)^2[1 + \sin^2 (3 \pi x_i+1)] + (x_n-1)^2 + \sin^2 (2 \pi x_n)) + \sum_{i=1}^{D} u(x_i, 5, 100, 4)$ | -50  | 50   | 30   |



### Discussions

| Average Best Value | GOA       | PGOA      |
| ------------------ | --------- | --------- |
| F1                 | 504.58    | 17.04     |
| F2                 | 6.97e+39  | 7.03e+37  |
| F3                 | -69503.89 | -87230.38 |
| F4                 | 5.96      | 2.52      |
| F5                 | 13891.70  | 971.43    |
| F6                 | 260.01    | 18.73     |
| F7                 | 14.38     | 3.07      |
| F8                 | -9863.94  | -14078.60 |
| F9                 | 992.35    | 765.23    |
| F10                | 5.46      | 1.64      |
| F11                | 1546.77   | 1345.37   |
| F12                | 32.52     | 5.58      |
| F13                | 12.87     | 2.55      |



| Average Running Time | GOA      | PGOA    |
| -------------------- | -------- | ------- |
| F1                   | 49.61s   | 13.05s  |
| F2                   | 77.01s   | 24.07s  |
| F3                   | 114.15s  | 24.64s  |
| F4                   | 48.91s   | 13.17s  |
| F5                   | 98.13s   | 23.28s  |
| F6                   | 55.33s   | 14.83s  |
| F7                   | 71.90s   | 18.95s  |
| F8                   | 58.54s   | 15.53s  |
| F9                   | 68.80s   | 18.05s  |
| F10                  | 109.09s  | 28.36s  |
| F11                  | 87.49s   | 22.75s  |
| F12                  | 187.64 s | 47.70 s |
| F13                  | 178.30 s | 45.38 s |



### Figures

<img src="/Users/sudo/Desktop/Research/src/figs/PGOA/cec_1_50d.png" alt="cec_1_50d" style="zoom:6%;" />

<img src="/Users/sudo/Desktop/Research/src/figs/PGOA/cec_2_50d.png" alt="cec_2_50d" style="zoom:6%;" />

<img src="/Users/sudo/Desktop/Research/src/figs/PGOA/cec_3_30d.png" alt="cec_3_30d" style="zoom:6%;" />

<img src="/Users/sudo/Desktop/Research/src/figs/PGOA/cec_4_50d.png" alt="cec_4_50d" style="zoom:6%;" />

<img src="/Users/sudo/Desktop/Research/src/figs/PGOA/cec_5_50d.png" alt="cec_5_50d" style="zoom:6%;" />

<img src="/Users/sudo/Desktop/Research/src/figs/PGOA/cec_6_50d.png" alt="cec_6_50d" style="zoom:6%;" />

<img src="/Users/sudo/Desktop/Research/src/figs/PGOA/cec_7_30d.png" alt="cec_7_30d" style="zoom:6%;" />

<img src="/Users/sudo/Desktop/Research/src/figs/PGOA/cec_8_50d.png" alt="cec_8_50d" style="zoom:6%;" />

<img src="/Users/sudo/Desktop/Research/src/figs/PGOA/cec_9_50d.png" alt="cec_9_50d" style="zoom:6%;" />

<img src="/Users/sudo/Desktop/Research/src/figs/PGOA/cec_10_50d.png" alt="cec_10_50d" style="zoom:6%;" />

<img src="/Users/sudo/Desktop/Research/src/figs/PGOA/cec_11_50d.png" alt="cec_11_50d" style="zoom:6%;" />

<img src="/Users/sudo/Desktop/Research/src/figs/PGOA/cec_12_30d.png" alt="cec_12_30d" style="zoom:6%;" />

<img src="/Users/sudo/Desktop/Research/src/figs/PGOA/cec_13_30d.png" alt="cec_13_30d" style="zoom:6%;" />