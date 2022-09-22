from sko.PSO import PSO
import numpy as np
import get_functions_details as func
import time
from datetime import datetime


now = datetime.now()
print("Starting Time: ", now.strftime("%Y-%m-%d %H:%M:%S"))

func, l_b, u_b, D = func.func('F1')

lb = np.ones(D) * l_b
ub = np.ones(D) * u_b

pso = PSO(func=func, n_dim=D, pop=160, max_iter=1000, lb=l_b, ub=u_b, w=0.8, c1=0.5, c2=0.5)
zone = time.time()
pso.run()
print('running time:', time.time() - zone)
print('best_x is ', pso.gbest_x, 'best_y is', pso.gbest_y)