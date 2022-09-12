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

"""
Rosenbrock  (strategy.1)
Rastrigin   (strategy.2)
Griewank    (strategy.3)
"""
func, l_b, u_b, D = func.func('Rastrigin')

lb = np.ones(D) * l_b
ub = np.ones(D) * u_b

# (population, dimension, max_iter, lb, ub, func)
goa = GOA.GOA(N, D, Iter, lb, ub, func)
zone_0 = time.time()
goa.run()
plt.plot(np.arange(goa.max_iter), goa.curve[1:], '-', label='GOA', linewidth=1)
print('GOA: ', time.time() - zone_0, 's', goa.best)

"""
groups: number of parallel groups (2^m, m is a positive integer)
strategy: parallel strategy choice (1, 2, 3)
migration: the rate of the best particle position is migrated and mutated to substitute the particles of the receiving group (0.25, 0.5, 0.75, 1.0)
copies: the number of worse particles substituted at each receiving group (1, 2)
communications: the number of iterations for communication
"""
groups = 8
strategy = 2
migration = 0.25
copies = 1
communications = 20

# (population, dimension, max_iter, lb, ub, func, groups, strategy, migration, copies, communications)
# pgoa = PGOA.PGOA(N, D, Iter, lb, ub, func, 2, strategy, migration, copies, communications)
# zone_1 = time.time()
# pgoa.run()
# plt.plot(np.arange(pgoa.max_iter), pgoa.curve[1:], '--', label='PGOA(2, 80)', linewidth=1)
# print('PGOA(2, 80): ', time.time() - zone_1, 's', pgoa.global_min)

# (population, dimension, max_iter, lb, ub, func, groups, strategy, migration, copies, communications)
pgoa = PGOA.PGOA(N, D, Iter, lb, ub, func, groups, strategy, migration, copies, communications)
zone_2 = time.time()
pgoa.run()
plt.plot(np.arange(pgoa.max_iter), pgoa.curve[1:], '--', label='PGOA', linewidth=1)
print('PGOA: ', time.time() - zone_2, 's', pgoa.global_min)

# (population, dimension, max_iter, lb, ub, func, groups, strategy, migration, copies, communications)
# pgoa = PGOA.PGOA(N, D, Iter, lb, ub, func, 8, strategy, migration, copies, communications)
# zone_3 = time.time()
# pgoa.run()
# plt.plot(np.arange(pgoa.max_iter), pgoa.curve[1:], '--', label='PGOA(8, 20)', linewidth=1)
# print('PGOA(8, 20): ', time.time() - zone_3, 's', pgoa.global_min)

# (population, dimension, max_iter, lb, ub, func, groups, strategy, migration, copies, communications)
# pgoa = PGOA.PGOA(N, D, Iter, lb, ub, func, groups, strategy, 1.0, copies, communications)
# zone_4 = time.time()
# pgoa.run()
# plt.plot(np.arange(pgoa.max_iter), pgoa.curve[1:], '--', label='PGOA: Migration=100%', linewidth=1)
# print('PGOA: Migration=100%: ', time.time() - zone_4, 's', pgoa.global_min)


plt.title('Strategy-2 PGOA(8, 20) Migration=25% Copies=1 100D')
mpl.rcParams.update({'font.size': 10})
plt.xlabel('Iteration')
plt.ylabel('Fitness Function Value')
plt.grid()
plt.legend()
mpl.rcParams.update({'font.size': 9})
plt.savefig('/Users/sudo/Desktop/Research/src/s2_g8_100d.png', dpi=1200)
plt.show()
