import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

import GOA as GOA
import PGOA as PGOA
import get_functions_details as func
import time
from datetime import datetime


now = datetime.now()
print("Starting Time: ", now.strftime("%Y-%m-%d %H:%M:%S"))

"""
Test cases
D: dimension
N: population
Iter: max iteration
func: fitness function
lb: lower bound
ub: upper bound
"""

N = 160
Iter = 100

runs = 50
running_time = np.zeros(4)
curve = np.zeros((4, Iter))
best = np.zeros(4)

"""
Rosenbrock  (strategy.1)
Rastrigin   (strategy.2)
Griewank    (strategy.3)
"""
func, l_b, u_b, D = func.func('Rosenbrock')

lb = np.ones(D) * l_b
ub = np.ones(D) * u_b

for iter in range(runs):

    # (population, dimension, max_iter, lb, ub, func)
    goa = GOA.GOA(N, D, Iter, lb, ub, func)
    zone_0 = time.time()
    goa.run()
    running_time[0] += time.time() - zone_0
    curve[0] += goa.curve[1:]
    best[0] += goa.best

    """
    groups: number of parallel groups (2^m, m is a positive integer, N/groups is an integer)
    strategy: parallel strategy choice (1, 2, 3)
    migration: the rate of the best particle position is migrated and mutated to substitute the particles of the receiving group (0.25, 0.5, 0.75)
    copies: the number of worse particles substituted at each receiving group (1, 2, 4)
    communications: the number of iterations for communication (Iter/communications is an integer)
    """
    groups = 8
    strategy = 1
    migration = 0.75
    copies = 1
    communications = 20

    # (population, dimension, max_iter, lb, ub, func, groups, strategy, migration, copies, communications)
    pgoa = PGOA.PGOA(N, D, Iter, lb, ub, func, groups, strategy, 0.25, copies, communications)
    zone_1 = time.time()
    pgoa.run()
    running_time[1] += time.time() - zone_1
    curve[1] += pgoa.curve[1:]
    best[1] += pgoa.global_min

    # (population, dimension, max_iter, lb, ub, func, groups, strategy, migration, copies, communications)
    pgoa = PGOA.PGOA(N, D, Iter, lb, ub, func, groups, strategy, 0.5, copies, communications)
    zone_2 = time.time()
    pgoa.run()
    running_time[2] += time.time() - zone_2
    curve[2] += pgoa.curve[1:]
    best[2] += pgoa.global_min

    # (population, dimension, max_iter, lb, ub, func, groups, strategy, migration, copies, communications)
    pgoa = PGOA.PGOA(N, D, Iter, lb, ub, func, groups, strategy, 0.75, copies, communications)
    zone_3 = time.time()
    pgoa.run()
    running_time[3] += time.time() - zone_3
    curve[3] += pgoa.curve[1:]
    best[3] += pgoa.global_min


# data processing
for i in range(4):
    running_time[i] = running_time[i] / runs
    curve[i] = curve[i] / runs
    best[i] = best[i] / runs

    # outcomes
    print('Experiment', i, ' Avg Best (', best[i], ') Avg T (', running_time[i], ')')

# plot
plt.plot(np.arange(Iter), curve[0], '-', label='GOA', linewidth=1)
plt.plot(np.arange(Iter), curve[1], '--', label='PGOA: Migration=25%', linewidth=1)
plt.plot(np.arange(Iter), curve[2], '--', label='PGOA: Migration=50%', linewidth=1)
plt.plot(np.arange(Iter), curve[3], '--', label='PGOA: Migration=75%', linewidth=1)

plt.title('Strategy-1 PGOA(8, 20) 100D')
mpl.rcParams.update({'font.size': 10})
plt.xlabel('Iteration')
plt.ylabel('Best Solution')
plt.grid()
plt.legend()
mpl.rcParams.update({'font.size': 9})
plt.savefig('/Users/sudo/Desktop/Research/src/figs/PGOA/s1_g8_100d.png', dpi=1200)
plt.show()
