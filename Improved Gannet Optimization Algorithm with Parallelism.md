# Improved Gannet Optimization Algorithm with Parallelism

<center><div style='height:2mm;'></div><div style="font-size:14pt;">Author: Jingbo Su</div></center>
<center><span style="font-size:9pt;line-height:9mm"><i>North China University of Technology</i></span>
</center>
<div> </br>
<div style="width:100px;float:left;line-height:14pt;font-size:15px"><b>Abstract: </b></div> 
<div style="overflow:hidden;line-height:14pt"></div>
</div> </br>
<div>
<div style="width:100px;float:left;line-height:14pt;font-size:15px"><b>Key Words: </b></div> 
<div style="overflow:hidden;line-height:14pt">Parallel Gannet Optimization Algorithm, Parallel Mechanisms, Communication Strategies, Rosenbrock, Rastrigin, and Griewank functions, CEC2013 benchmark functions.</div>
</div>


This paper introduces a novel improvement of the Gannet Optimization Algorithm(GOA) using parallelism. Nowadays, algorithms are inevitably used to solve a large number of high-dimensional complex problems in numerous fields. Although there are many meta-heuristics algorithms such as GOA, as a meta-heuristics algorithm, which is good at solving these high-dimensional com

perform better than previous exact methods such as dynamic programming when dealing with these problems, they sometimes run very long and may get stuck in local optima. GOA as a meta-heuristics algorithm, which is good at solving 

In this paper, the Parallel Gannet Optimization Algorithm(PGOA) using the combination of global and group communication strategies is purposed. Compared with the original GOA, the improved algorithm involved in this paper not only has better convergence results but effectively shortens the running time.

## Introduction

### Meta-heuristic Algorithms

Meta-heuristic algorithms, i.e., optimization methods designed according to the strategies laid out in a meta-heuristic framework, are — as the name suggests — always heuristic in nature[1]. Some popular meta-heuristic algorithms are Particle Swarm Optimization(PSO)[2], Ant Colony Optimization(ACO)[3], Cat Swarm Optimization(CSO)[4], etc. There are so many meta-heuristic algorithms designed based on the activities of creatures in nature. And especially for complex problems with high dimensions or large scales, the results of the combination of meta-heuristic and swarm intelligence optimization sometimes perform well. As a result, meta-heuristic algorithms can be a viable alternative to most exact methods such as branch and bound and dynamic programming.

### Parallel Mechanisms

The purpose of parallel processing is concerned with running the same algorithm or program simultaneously in the macro perspective in order to reduce the running time. The two basic parallel processing methods are pipeline processing and data parallelism[5]. And the communication between groups(i.e. replacing worse solutions from other groups) can accelerate the convergence speed. The parallelization strategy outperforms the original algorithms because it is easier to implement multi-parallel searches in the space and create more communication between groups than the fundamental population-based algorithms. Therefore, the parallelization strategy is widely used in many fields such as data mining[6] and deep learning[7], to solve many complicated and large-scale engineering problems.

### Motivation

As data proliferates and becomes more complicated, a single communication strategy is often not sufficient. For example, the GA[8], as one of the earliest algorithms employed a parallel strategy. The parallel strategy in the PGA is used for simulating biological mating so that children are allowed to select their parents randomly. It helps to improve the GA, however, the mating operation is largely independent of any other[9]. Therefore, a single communication strategy is liable to cause the local optima during iterations. Roddick et al. Three communication strategies are thus proposed and applied to PSO. Experiments demonstrated that the improved strategy indeed successfully improved the efficiency of the PSO[2]. Although the parallel strategy purposed in this paper will be based on Roddick et al.'s strategy, strategy-1, which uses the global best solution to update the current group solution, the algorithm is easily prone to trap into the local optima. Eventually, it influence strategy-3 as well. The same problem affects the performance of PCWOA[10] as well, on the other hand, the number of groups in PCWOA is too small to guarantee adequate communication.

In this paper, we discuss the application of parallel strategy to improve the GOA. In the next section, the basic idea of GOA will be described. Section 3 presents the application of parallel strategy in the GOA. Section 4 demonstrates the merits of the PGOA through plenty of experiments. Section 5 summarizes the experimental outcomes.

## Gannet Optimization Algorithm

The Gannet Optimization Algorithm, as a new nature-inspired meta-heuristic algorithm, mathematizes the various unique behaviors of gannets during foraging and is used to enable exploration and exploitation[11]. GOA is a population-based meta-heuristics algorithm(such as WOA[12] and ROA[13]) and thus it has a host of similarities(including individual position matrix, etc.) with them. However, thanks to GOA's U-shaped and V-shaped diving patterns(during the exploration phase), it is possible to explore the optimal region within the search space, where sudden turns and random walks can ensure better solutions are found. In addition, as the dimension increases, experiments disclose that the GOA not only shows significantly superior results but also is more effective(less running time) than other algorithms. Due to the competitive advantages of GOA in high dimensions, we decide to improve it to solve high-dimensional problems quicker.

### Initialization phase

GOA processes the search scheme using a position matrix X:
$$
X = 
\begin{bmatrix}
x_{1,1} & x_{1,2} & \cdots & x_{1,D} \\
x_{2,1} & x_{2,2} & \cdots & x_{2,D} \\
\vdots  & \vdots  & \ddots & \vdots  \\
x_{N,1} & x_{N,2} & \cdots & x_{N,D} 
\end{bmatrix}
$$
$x_i$ , a particle with D dimensions, denotes the position of the i-th individual. And each individual can be calculated by $x_{i,j}=r_1\cdot (ub_j-lb_j)+lb_j,\ i=1,2,\dots,N,\ j=1,2,\dots,D$ is equivalent to a candidate solution to the problem.

$r$ is a random number uniformly distributed between 0 and 1, thus the chances of exploration and exploitation phases are equal.

### Exploration phase

As gannets find prey, they adjust their dive pattern in terms of the depth of the prey diving. There are two types of diving are purposed: a long and deep U-shaped dive and a short and shallow V-shaped dive[14]:
$$
V(x) =
\begin{cases}
	-\displaystyle \frac{1}{\pi} \cdot x + 1,\ x \in (0, \pi)\\\\
	\displaystyle \frac{1}{\pi} \cdot x - 1,\ x \in (\pi, 2\pi)
\end{cases}
\\\\
t = 1 - \displaystyle \frac{Iter}{T_{max\_iter}}\\\\
a = 2 \cdot \cos(2\pi r_2) \cdot t\\\\
b = 2 \cdot V(2\pi r_3) \cdot t\\\\
A = (2r_4 - 1) \cdot a\\\\
B = (2r_5 - 1) \cdot b\\\\
X_m(t) = \displaystyle \frac{1}{N} \sum_{i=1}^N X_i(t)\\\\
u_2 = A \cdot (X_i(t) - X_r(t))\\\\
v_2 = B \cdot (X_i(t) - X_m(t))\\\\
MX_i(t + 1) =
\begin{cases}
	X_i(t) + u_1 + u_2,\ q \ge 0.5\\\\
	X_i(t) + v_1 + v_2,\ q < 0.5
\end{cases}
$$
$q$ and all $r_i$ are random values ranging from 0 to 1; The gannet will behave in a U-shaped dive pattern if $q \ge 0.5$ , and use a V-shaped pattern otherwise.

### Exploitation phase

Two actions are proposed when the gannet captures prey after rushing into the water-Levy and Turns. Here they define a variable called Capture Capacity, which is primarily affected by the energy of the gannet. If the gannets have sufficient energy, they will perform a Levy random walk, otherwise, in most cases, Turn behavior is common when catching prey:
$$
L = 0.2 + (2 - 0.2) \cdot r_6\\\\
R = \displaystyle \frac{M\cdot velocity^2}{L}\ (M=2.5kg,\ velocity=1.5m/s)\\\\
t_2 = 1 + \displaystyle \frac{Iter}{T_{max\_iter}}\\\\
Capturability = \displaystyle \frac{1}{R \cdot t_2}\\\\
\sigma = \displaystyle (\frac{\Gamma(1 + \beta) \cdot \sin(\frac{\pi \beta}{2})}{\Gamma(\frac{1 + \beta}{2} \cdot \beta \cdot 2^{\frac{\beta - 1}{2}})})^{\frac{1}{\beta}}\\\\
Levy(Dim) = 0.01 \cdot \displaystyle \frac{\mu \cdot \sigma}{|v|^{\frac{1}{\beta}}}\\\\
P = Levy(Dim)\\\\
delta = Captureability \cdot |X_i(t)-X_{best}(t)|\\\\
MX_i(t+1) = 
\begin{cases}
	t \cdot delta \cdot (X_i(t) - X_{best}(t)) + X_i(t)\ \text{ , Capturability}\ge c\\\\
	X_{best} - (X_i(t) - X_{best}) \cdot P \cdot t\ \text{ , Capturability}< c\ (c = 0.2)
\end{cases}
$$
$c$ and $r_6$ are random values ranging from 0 to 1; The gannet will perform Levy walk if $Captureability \ge c (c=0.2)$ , and reveal Turn walk otherwise.

## Parallel Gannet Optimization Algorithm

- Strategy-1:

    Migrate the global best(the best value among all individuals) to substitute the worse individuals in each group for every $2T$ iterations. And migrate the group best(the best value among the current group) to substitute the worse individuals in the current group for every $T$ iterations.

- Strategy-2:

    Migrate the best individual of the p-th group to the q-th group to replace the worse solutions just like Strategy-1 does for every $T$ iterations. Here $q = p\  \newcommand*\xor{\oplus}\ 2^m,\ p=0, \dots,G-1,\ m=0,\dots,\ n-1$ and $G = 2^n$, n is a positive integer.

- Strategy-3:

    Divide these groups into two subgroups of an equal number of individuals. Apply Strategy-1 and Strategy-2 with equal probability respectively.

In this article, more groups are created to participate in communication. Compared with the number of groups in PPSO[5], it is 2, 4, and 8; the group number in PCWOA[10] is 3, and for the purpose of enhancing the randomness of communication, we determine to increase the number of groups to 16, and experiments demonstrate the improved efficiency. 

Strategy-2 is stochastic communication between groups and performs well in the Rastrigin Function. Therefore, local search such as intra-group communication helps to find the optimal solution while avoiding stuck in local optima(It can be seen in the following experiments). Specifically, in Strategy-1 we no longer substitute the worse individuals with the global best every $T$ iterations, but extend the global replacement period to $2T$, and exchange every $T$ iterations use local substitution.

***Additional tips are as follows:***

- The fitness values of individuals in each group are sorted in descending order to substitute the worse particles with the given migration rate.

- The migration rate in Strategy-1 is 25%, 50%, and 75%(the 100% migration rate loses the randomness of search).

- The copy time in Strategy-2 is 1, 2, and 4(increase random intra-group communications).

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
      do exploration()
    else:
      do exploitation()
    for i in Np:
      Update G[g].pop_fit[i], G[g].best, G[g].min, global_best, global_min
    if St is 1:
      do Strategy-1
    elif St is 2:
      do Strategy-2
    else:
      do Strategy-3
```

### Parameter Tuning

The first and second experiment is conducted to test the performance of the first PGOA communication strategy in the 30-D Rosenbrock and Rastrigin function. And the number of iterations and communications are set to 50 and 10, respectively. The third experiment aims to test the performance of the second strategy in the 30-D Griewank function. And the number of iterations and the number of communications are set to 100 and 20. To be fair, each of the three experiments is performed 50 times and averaged; and the total control population is constantly 160.

#### First Experiment

Better solutions are migrated to substitute worse individuals at 25%, 50%, and 75%(we give up 100% replacement as it may result in loss of randomness).

<img src="/Users/sudo/Desktop/Research/src/figs/PGOA/s1_f1_30d_mt.png" alt="s1_f1_30d_mt" style="zoom:6%;" />

| 50 Runs Avg    | GOA      | PGOA(M=25%) | PGOA(M=50%) | PGOA(M=75%) |
| :------------- | -------- | ----------- | ----------- | ----------- |
| Best Solution  | 30195.35 | 438.10      | 480.17      | 305.37      |
| Running Time/s | 37.01    | 5.02        | 4.99        | 4.98        |

#### Second Experiment

The substitution rate is controlled at 75% and the entire population was divided into 4, 8, and 16 groups to examine the best grouping strategy.

<img src="/Users/sudo/Desktop/Research/src/figs/PGOA/s1_f2_30d_gt.png" alt="s1_f2_30d_gt" style="zoom:6%;" />

| 50 Runs Avg    | GOA    | PGOA(G=4) | PGOA(G=8) | PGOA(G=16) |
| :------------- | ------ | --------- | --------- | ---------- |
| Best Solution  | 545.62 | 456.79    | 466.29    | 469.81     |
| Running Time/s | 32.92  | 8.67      | 4.57      | 2.43       |

#### Third Experiment

Both 75% migration and 4 groups are set based on the result of experiment 1 and 2. The times of copy is set to 1, 2, and 4 to find a better times of copy.

<img src="/Users/sudo/Desktop/Research/src/figs/PGOA/s2_f3_30d_ct1.png" alt="s2_f3_30d_ct1" style="zoom:6%;" />

| 50 Runs Avg    | GOA    | PGOA(C=1) | PGOA(C=2) | PGOA(C=4) |
| :------------- | ------ | --------- | --------- | --------- |
| Best Solution  | 831.73 | 766.19    | 689.23    | 696.97    |
| Running Time/s | 84.48  | 21.80     | 21.99     | 21.89     |



### Final Experiments

In the last set of experiments,  13 of CEC2013 benchmark functions are chosen as test functions to compare the performance of PGOA and GOA. The parameters for test are displayed as following:

| Parameter     | Value |
| :------------ | ----: |
| Execution     |    50 |
| Iteration     |   100 |
| Population    |   160 |
| Strategy      |     3 |
| Communication |    20 |
| Migration     |   75% |
| Group         |     4 |
| Copy          |     2 |



### Uni-modal Test Functions

|  ID  | Function                                                     |  LB   |  UB  | Dim  |
| :--: | :----------------------------------------------------------- | :---: | :--: | :--: |
|  F1  | $f(x) = \displaystyle \sum_{i=1}^D x_i^2$                    | -100  | 100  |  50  |
|  F2  | $f(x) = \displaystyle \sum_{i=1}^D\ \lvert x_i \lvert + \prod_{i=1}^D\  \lvert x_i \lvert$ |   5   |  10  |  50  |
|  F3  | $f(x) = \displaystyle \sum_{i=1}^D (\sum_{j=1}^i x_j)$       | -100  | 100  |  30  |
|  F4  | $f(x) = \max \{\lvert x_i \lvert ,\ 1 \leq i \leq n \}$      | -100  | 100  |  50  |
|  F5  | $f(x) = \displaystyle \sum_{i=1}^{D-1}\ [100(x_{i+1}-x_i^2)^2+(x_i-1)^2]$ |  -5   |  30  |  50  |
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



## Conclusions



## References

[1] Metaheuristic

[2] PSO

[3] ACO

[4] CSO

[5] PPSO

[6] Data Mining

[7] Deep Learning

[8] GA

[9] PGA

[10] PCWOA

[11] GOA

[12] WOA

[13] ROA

[14] U-shaped & V-shaped
