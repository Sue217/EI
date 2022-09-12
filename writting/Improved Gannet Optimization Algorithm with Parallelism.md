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



**Claim**	Improvement of the PGOA & Good performance



**Reason.1**	PGOA has a better performance in high dimensions

**Evidence.1**	GOA can provide a better solution in high dimensions (than other evolutionary algorithms), however, PGOA performs better (faster convergence) than GOA in high dimensions (30D, 50D, 100D)



**Reason.2**	PGOA has a shorter running time in high dimensions

**Evidence.2**	GOA has a shorter running time in high dimensions compared with other comparison algorithms, however, PGOA runs faster than GOA in high dimensions (30D, 50D, 100D)



**Reason.3**	

**Evidence.3**	



**Reason.4** 

**Evidence.4** 



## Introduction

Meta-heuristic algorithms, i.e., optimization methods designed according to the strategies laid out in a metaheuristic framework, are — as the name suggests — always heuristic in nature[1]. As a matter of fact, there are so many meta-heuristic algorithms are designed based on the activities of swarm in the nature. And especially for complex problems with high dimensions or large scales, the results of the combination of meta-heuristic and swarm intellgence optimization sometimes perform well. As a result, meta-heuristic algorithms are able to be a viable alternative to most exact methods such as branch and bound and dynamic programming. 

The Gannet Optimization Algorithm, as a new nature-inspired meta-heuristic algorithm, mathematizes the various unique behaviors of gannets during foraging and is used to enable exploration and exploitation[2]. GOA is a population based meta-heuristics algorithm(such as PSO[3] and WOA[4]) and thus it has a host of samilarities(includes individual position martrix, etc.) with them. However, thanks to GOA's U-shaped and V-shaped diving patterns(during the exploration phase), it is possible to explore the optimal region within the search space, where sudden turns and random walks can ensure the better solutions are found.

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
$x_i$ , an particle with D dimensions, denotes the position of the i-th individual. And each individual that can be calculated by $x_{i,j}=r_1\cdot (ub_j-lb_j)+lb_j,\ i=1,2,\dots,N,\ j=1,2,\dots,D$ is equivalent to a candidate solution of the problem.