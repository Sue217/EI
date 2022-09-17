# Improved Gannet Optimization Algorithm with Parallelism

<center><div style='height:2mm;'></div><div style="font-size:14pt;">Author: Jingbo Su</div></center>
<center><span style="font-size:9pt;line-height:9mm"><i>North China University of Technology</i></span>
</center>
<div> </br>
<div style="width:100px;float:left;line-height:14pt;font-size:15px"><b>Abstract: </b></div> 
<div style="overflow:hidden;line-height:14pt">This paper introduces a novel improvement of the Gannet Optimization Algorithm(GOA) using parallelism, Parallel Gannet Optimization Algorithm(PGOA) is purposed. Nowadays, algorithms are inevitably used to solve a large number of high-dimensional complex problems in numerous domains. Although there are many meta-heuristics (such as GOA) that perform better than previous exact methods such as dynamic programming when dealing with these problems, they sometimes run very long and even get stuck in local optima. [Methods...][Results...] Compared with the original GOA, the improved algorithm involved in this paper not only has better convergence results but effectively shortens the running time.[Conclusion...]</div>
</div> </br>
<div>
<div style="width:100px;float:left;line-height:14pt;font-size:15px"><b>Key Words: </b></div> 
<div style="overflow:hidden;line-height:14pt">Parallel Gannet Optimization Algorithm, Communication Strategies, Rosenbrock, Rastrigrin, and Griewank functions.</div>
</div>


> **Claim**	Improvement of the PGOA & Good performance
>
> **Reason.1**	PGOA has a better performance in high dimensions
>
> **Evidence.1**	GOA can provide a better solution in high dimensions (than other evolutionary algorithms), however, PGOA performs better (faster convergence) than GOA in high dimensions (30D, 50D, 100D)
>
> **Reason.2**	PGOA has a shorter running time in high dimensions
>
> **Evidence.2**	GOA has a shorter running time in high dimensions compared with other comparison algorithms, however, PGOA runs faster than GOA in high dimensions (30D, 50D, 100D)



## Introduction

### Meta-heuristic Algorithms

Meta-heuristic algorithms, i.e., optimization methods designed according to the strategies laid out in a meta-heuristic framework, are — as the name suggests — always heuristic in nature[1]. Some popular meta-heuristic algorithms are Particle Swarm Optimization(PSO)[ ], Ant Colony Optimization(ACO)[ ], Cat Swarm Optimization(CSO)[ ], etc. There are so many meta-heuristic algorithms designed based on the activities of creatures in nature. And especially for complex problems with high dimensions or large scales, the results of the combination of meta-heuristic and swarm intelligence optimization sometimes perform well. As a result, meta-heuristic algorithms can be a viable alternative to most exact methods such as branch and bound and dynamic programming.

### Parallel Mechanisms

The purpose of parallel processing is concerned with running the same algorithm or program simultaneously in the macro perspective in order to reduce the running time. The two basic parallel processing methods are pipeline processing and data parallelism[ ]. And the communication between groups(i.e. replacing worse solutions from other groups) can accelerate the convergence speed. 

> Thanks to the development of hardware, people can afford multiple CPUs, which speeds up the search. Not only that, the parallelization strategy outperforms the original algorithms because it is easier to implement multi-parallel searches in space and create more communication between groups than the original population-based algorithms.

> Therefore, the parallelization strategy is widely used in many domains such as data mining[ ] and deep learning[ ], to solve many complicated and large-scale engineering problems.

### Motivation

As data proliferates and becomes more complicated, a single communication strategy is often not sufficient. For example, the GA[ ], as one of the earliest algorithms employed a parallel strategy. The parallel strategy in the PGA is used for simulating biological mating so that children are allowed to select their parents randomly. It helps to improve the GA, however, the mating operation is largely independent of any other[ ]. Therefore, a single communication strategy may easily cause the local optima during iterations. Roddick et al. Three communication strategies are thus proposed and applied to PSO. Experiments demonstrated that the improved strategy indeed successfully improved the efficiency of the PSO[ ]. Although the parallel strategy purposed in this paper will be based on Roddick et al.'s strategy, strategy-1, which uses the global best solution to update the current group solution, may easily make the algorithm trap into the local optima. Eventually, it may influence strategy-3 as well. The same problem affects the performance of PCWOA[ ] as well, on the other hand, the number of groups in PCWOA is too small to guarantee adequate communication.

In this paper, we discuss the application of parallel strategy to improve the GOA. In the next section, the basic idea of GOA will be described. Section 3 presents the application of parallel strategy in the GOA. Section 4 demonstrates the merits of the PGOA through plenty of experiments. Section 5 summarizes the experimental outcomes.

## Gannet Optimization Algorithm

> **(What & Why)**

The Gannet Optimization Algorithm, as a new nature-inspired meta-heuristic algorithm, mathematizes the various unique behaviors of gannets during foraging and is used to enable exploration and exploitation[ ]. GOA is a population-based meta-heuristics algorithm(such as WOA[ ] and ROA[ ]) and thus it has a host of similarities(including individual position matrix, etc.) with them. However, thanks to GOA's U-shaped and V-shaped diving patterns(during the exploration phase), it is possible to explore the optimal region within the search space, where sudden turns and random walks can ensure better solutions are found. In addition, as the dimension increases, experiments disclose that the GOA not only shows significantly superior results but also is more effective(less running time) than other algorithms. Due to the competitive advantages of GOA in high dimensions, we decide to improve it to solve high-dimensional problems quicker.

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

As gannets find prey, they adjust their dive pattern in terms of the depth of the prey diving. There are two types of diving are purposed: a long and deep U-shaped dive and a short and shallow V-shaped dive[ ]:
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

> **(How?)**



## Experiments



## Conclusions





## Reference