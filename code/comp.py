# author: Jingbo Su
# Comparison of GOA, CGOA and PCGOA

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import gamma
from scipy.stats import truncnorm
import time


class GOA:
    """
    population: population size
    dimension: problem dimension
    max_iter: maximum number of iterations
    lb, ub: lower bound, upper bound
    func: fitness function
    """

    def __init__(self, population, dimension, max_iter, lb, ub, func):
        self.population = population
        self.dimension = dimension
        self.max_iter = max_iter
        self.lb = lb
        self.ub = ub
        self.X = None
        self.MX = None
        self.Xb = None
        self.pop_fit = np.ones(self.population) * np.inf
        self.best = np.inf
        self.curve = np.zeros(self.max_iter)
        self.fitness_func = func

    def initialize(self):
        # X: D * N
        self.X = np.random.rand(self.dimension, self.population)
        if self.dimension == 1:
            self.X = np.random.rand(1, self.population) * (self.ub - self.lb) + self.lb
        elif self.dimension > 1:
            for d in range(self.dimension):
                self.X[d] = np.random.rand(1, self.population) * (self.ub[d] - self.lb[d]) + self.lb[d]

    def exploration(self, iteration):
        t = 1 - iteration / self.max_iter
        a = 2 * np.cos(2 * np.pi * np.random.rand()) * t

        def V(x):
            return ((-1 / np.pi) * x + 1) * (0 < x < np.pi) + ((1 / np.pi) * x - 1) * (np.pi <= x < 2 * np.pi)

        b = 2 * V(2 * np.pi * np.random.rand()) * t
        A = (2 * np.random.rand() - 1) * a
        B = (2 * np.random.rand() - 1) * b
        for iter in range(self.population):
            q = np.random.rand()
            Xi = self.X[:, iter]
            if q >= 0.5:
                u1 = np.random.uniform(-a, a, self.dimension)
                Xr = self.X[:, int(np.random.uniform(0, self.population, 1))]
                u2 = A * (Xi - Xr)
                self.MX[:, iter] = Xi + u1 + u2
            else:
                v1 = np.random.uniform(-b, b, self.dimension)
                Xm = np.mean(self.X)
                v2 = B * (Xi - Xm)
                self.MX[:, iter] = Xi + v1 + v2

            self.bound_check()
            self.update()

    def exploitation(self, iteration):
        t2 = 1 + iteration / self.max_iter
        M = 2.5
        vel = 1.5
        L = 0.2 + (2 - 0.2) * np.random.rand()
        R = (M * vel ** 2) / L
        Capturability = 1 / (R * t2)
        c = 0.2  # 0.15
        for iter in range(self.population):
            Xi = self.X[:, iter]
            if Capturability >= c:
                delta = Capturability * np.abs(Xi - self.Xb)
                self.MX[:, iter] = t2 * delta * (Xi - self.Xb) + Xi
            else:
                P = self.Levy(self.dimension)
                self.MX[:, iter] = self.Xb - (Xi - self.Xb) * P * t2

            self.bound_check()
            self.update()

    @staticmethod
    def Levy(dimension):
        beta = 1.5
        sigma = (gamma(1 + beta) * np.sin(np.pi * beta / 2) / (
            gamma(((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2)))) ** (1 / beta)
        mu = np.random.rand(1, dimension)
        v = np.random.rand(1, dimension)
        return 0.01 * mu * sigma / ((np.abs(v)) ** (1 / beta))

    def bound_check(self):
        for i in range(self.population):
            self.MX[:, i] = np.where(self.MX[:, i] < self.lb, self.lb, self.MX[:, i])
            self.MX[:, i] = np.where(self.MX[:, i] > self.ub, self.ub, self.MX[:, i])

    def update(self):
        for i in range(self.population):
            new_fit = self.fitness_func(self.MX[:, i])
            if new_fit < self.pop_fit[i]:
                self.pop_fit[i] = new_fit
                self.X[:, i] = self.MX[:, i]
            if new_fit < self.best:
                self.best = new_fit
                self.Xb = self.MX[:, i]

    def run(self):
        self.initialize()
        self.MX = self.X
        self.Xb = self.X

        for i in range(self.population):
            self.pop_fit[i] = self.fitness_func(self.X[:, i])
            if self.pop_fit[i] < self.best:
                self.best = self.pop_fit[i]
                self.Xb = self.X[:, i]

        for iteration in range(self.max_iter):
            rand = np.random.rand()
            if rand > 0.5:
                self.exploration(iteration)
            else:
                self.exploitation(iteration)
            self.curve[iteration] = self.best

        # plot
        plt.xlabel('iteration')
        plt.ylabel('best value')
        plt.plot(np.arange(self.max_iter), self.curve, label='GOA')


class CGOA:
    """
    population: population size
    dimension: problem dimension
    max_iter: maximum number of iterations
    lb, ub: lower bound, upper bound
    func: fitness function
    """

    def __init__(self, population, dimension, max_iter, lb, ub, func):
        self.population = population
        self.dimension = dimension
        self.max_iter = max_iter
        self.lb = lb
        self.ub = ub
        self.X = None
        self.MX = None
        self.pop_fit = np.inf
        self.curve = np.zeros(self.max_iter)
        self.fitness_func = func

        self.mu = np.zeros(self.dimension)
        self.sigma = np.ones(self.dimension) * 10.0
        self.Np = self.population
        self.best = self.ub
        self.fmin = np.inf

    def initialize(self):
        self.X = np.random.rand(self.dimension)
        for d in range(self.dimension):
            self.X[d] = np.random.rand() * (self.ub[d] - self.lb[d]) + self.lb[d]

    def exploration(self, iteration):
        t = 1 - iteration / self.max_iter
        a = 2 * np.cos(2 * np.pi * np.random.rand()) * t

        def V(x):
            return ((-1 / np.pi) * x + 1) * (0 < x < np.pi) + ((1 / np.pi) * x - 1) * (np.pi <= x < 2 * np.pi)

        b = 2 * V(2 * np.pi * np.random.rand()) * t
        A = (2 * np.random.rand() - 1) * a
        B = (2 * np.random.rand() - 1) * b

        q = np.random.rand()
        x1 = self.generate()

        if q >= 0.5:
            u1 = np.random.uniform(-a, a, self.dimension)
            Xr = self.generate()  # just have a try
            u2 = A * (x1 - Xr)
            self.MX = x1 + u1 + u2
        else:
            v1 = np.random.uniform(-b, b, self.dimension)
            Xm = np.mean(self.X)
            v2 = B * (x1 - Xm)
            self.MX = x1 + v1 + v2

        self.bound_check()
        self.update()
        x2 = self.X
        return self.compete(x1, x2)

    def exploitation(self, iteration):
        t2 = 1 + iteration / self.max_iter
        M = 2.5
        vel = 1.5
        L = 0.2 + (2 - 0.2) * np.random.rand()
        R = (M * vel ** 2) / L
        Capturability = 1 / (R * t2)
        c = 0.2

        x1 = self.generate()
        if Capturability >= c:
            delta = Capturability * (np.abs(x1 - self.best))
            self.MX = t2 * delta * (x1 - self.best) + x1
        else:
            P = self.Levy(self.dimension)
            self.MX = self.best - (x1 - self.best) * P * t2

        self.bound_check()
        self.update()
        x2 = self.X
        return self.compete(x1, x2)

    @staticmethod
    def Levy(dimension):
        beta = 1.5
        sigma = (gamma(1 + beta) * np.sin(np.pi * beta / 2) / (
            gamma(((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2)))) ** (1 / beta)
        mu = np.random.rand(dimension)
        v = np.random.rand(dimension)
        return 0.01 * mu * sigma / ((np.abs(v)) ** (1 / beta))

    def bound_check(self):
        self.MX = np.where(self.MX < self.lb, self.lb, self.MX)
        self.MX = np.where(self.MX > self.ub, self.ub, self.MX)

    def generate(self):
        # bug causing...

        x = np.random.rand(self.dimension)
        a, b = (-1 - self.mu) / self.sigma, (1 - self.mu) / self.sigma
        y = truncnorm.cdf(x, a, b, loc=self.mu, scale=self.sigma)
        y = np.where(np.isnan(y), 1e-20, y)
        x_actual = y * (self.ub - self.lb) / 2 + (self.ub + self.lb) / 2
        return x_actual

    def compete(self, x1, x2):
        fx1 = self.fitness_func(x1)
        fx2 = self.fitness_func(x2)
        if fx1 < fx2:
            winner, loser = fx1, fx2
        else:
            winner, loser = fx2, fx1
        return winner, loser

    def update(self):
        new_fit = self.fitness_func(self.MX)
        if new_fit < self.pop_fit:
            self.pop_fit = new_fit
            self.X = self.MX
        if new_fit < self.fmin:
            self.fmin = new_fit
            self.best = self.MX

    def run(self):
        self.initialize()
        self.MX = self.X
        self.pop_fit = self.fitness_func(self.X)
        if self.pop_fit < self.fmin:
            self.fmin = self.pop_fit
            self.best = self.X

        for iteration in range(self.max_iter):
            rand = np.random.rand()
            if rand > 0.5:
                winner, loser = self.exploration(iteration)
            else:
                winner, loser = self.exploitation(iteration)

            last = self.mu
            self.mu = last + (1 / self.Np) * np.abs(winner - loser)
            self.sigma = np.sqrt(np.square(self.sigma) + np.square(last) - np.square(self.mu) + (1 / self.Np) * np.abs(np.square(winner) - np.square(loser)))

            # update
            winner, loser = self.compete(self.X, self.best)
            self.best = winner
            self.curve[iteration] = self.fitness_func(self.best)

        # plot
        plt.xlabel('$Iterations$')
        plt.ylabel('$Best$ $value$')
        plt.plot(np.arange(self.max_iter), self.curve, label='CGOA')


class PCGOA:
    """
    population: population size
    dimension: problem dimension
    max_iter: maximum number of iterations
    lb, ub: lower bound, upper bound
    func: fitness function
    groups: number of parallel groups
    """

    def __init__(self, population, dimension, max_iter, lb, ub, func, groups=3):
        self.population = population
        self.dimension = dimension
        self.max_iter = max_iter
        self.lb = lb
        self.ub = ub
        self.X = None
        self.MX = None
        self.curve = np.zeros(self.max_iter)
        self.fitness_func = func

        self.groups = groups
        self.pop_fit = np.ones(self.groups) * np.inf

        self.mu = np.zeros((self.groups, self.dimension))
        self.sigma = np.ones((self.groups, self.dimension)) * 10.0
        self.Np = self.population / self.groups
        self.best = np.ones((self.groups, 1)) * self.ub
        self.fmin = np.ones(self.groups) * np.inf

        self.global_best = np.inf
        self.global_fmin = np.inf
        self.total_best = np.inf

    def initialize(self):
        self.X = np.random.rand(self.groups, self.dimension)
        for d in range(self.dimension):
            self.X[:, d] = np.random.rand() * (self.ub[d] - self.lb[d]) + self.lb[d]

    def exploration(self, iteration, group):
        t = 1 - iteration / self.max_iter
        a = 2 * np.cos(2 * np.pi * np.random.rand()) * t

        def V(x):
            return ((-1 / np.pi) * x + 1) * (0 < x < np.pi) + ((1 / np.pi) * x - 1) * (np.pi <= x < 2 * np.pi)

        b = 2 * V(2 * np.pi * np.random.rand()) * t
        A = (2 * np.random.rand() - 1) * a
        B = (2 * np.random.rand() - 1) * b

        q = np.random.rand()
        x1 = self.generate(group)

        if q >= 0.5:
            u1 = np.random.uniform(-a, a, self.dimension)
            Xr = self.generate(group)  # just have a try
            u2 = A * (x1 - Xr)
            self.MX[group] = x1 + u1 + u2
        else:
            v1 = np.random.uniform(-b, b, self.dimension)
            Xm = np.mean(self.X)
            v2 = B * (x1 - Xm)
            self.MX[group] = x1 + v1 + v2

        self.bound_check(group)
        self.update(group)
        x2 = self.X[group]
        return self.compete(x1, x2)

    def exploitation(self, iteration, group):
        t2 = 1 + iteration / self.max_iter
        M = 2.5
        vel = 1.5
        L = 0.2 + (2 - 0.2) * np.random.rand()
        R = (M * vel ** 2) / L
        Capturability = 1 / (R * t2)
        c = 0.2

        x1 = self.generate(group)
        if Capturability >= c:
            delta = Capturability * (np.abs(x1 - self.best[group]))
            self.MX[group] = t2 * delta * (x1 - self.best[group]) + x1
        else:
            P = self.Levy(self.dimension)
            self.MX[group] = self.best[group] - (x1 - self.best[group]) * P * t2

        self.bound_check(group)
        self.update(group)
        x2 = self.X[group]
        return self.compete(x1, x2)

    @staticmethod
    def Levy(dimension):
        beta = 1.5
        sigma = (gamma(1 + beta) * np.sin(np.pi * beta / 2) / (
            gamma(((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2)))) ** (1 / beta)
        mu = np.random.rand(1, dimension)
        v = np.random.rand(1, dimension)
        return 0.01 * mu * sigma / ((np.abs(v)) ** (1 / beta))

    def bound_check(self, group):
        self.MX[group] = np.where(self.MX[group] < self.lb, self.lb, self.MX[group])
        self.MX[group] = np.where(self.MX[group] > self.ub, self.ub, self.MX[group])

    def generate(self, group):
        # bug causing...

        x = np.random.rand(self.dimension)
        a, b = (-1 - self.mu[group]) / self.sigma[group], (1 - self.mu[group]) / self.sigma[group]
        y = truncnorm.cdf(x, a, b, loc=self.mu[group], scale=self.sigma[group])
        y = np.where(np.isnan(y), 1e-20, y)
        x_actual = y * (self.ub - self.lb) / 2 + (self.ub + self.lb) / 2
        return x_actual

    def compete(self, x1, x2):
        fx1 = self.fitness_func(x1)
        fx2 = self.fitness_func(x2)
        if fx1 < fx2:
            winner, loser = fx1, fx2
        else:
            winner, loser = fx2, fx1
        return winner, loser

    def update(self, group):
        new_fit = self.fitness_func(self.MX[group])
        if new_fit < self.pop_fit[group]:
            self.pop_fit[group] = new_fit
            self.X[group] = self.MX[group]
        if new_fit < self.fmin[group]:
            self.fmin[group] = new_fit
            self.best[group] = self.MX[group]

    def run(self):
        self.initialize()
        self.MX = self.X
        for iteration in range(self.max_iter):
            for g in range(self.groups):
                self.pop_fit[g] = self.fitness_func(self.X[g])
                if self.pop_fit[g] < self.fmin[g]:
                    self.fmin[g] = self.pop_fit[g]
                    self.best[g] = self.X[g]

                rand = np.random.rand()
                if rand > 0.5:
                    winner, loser = self.exploration(iteration, g)
                else:
                    winner, loser = self.exploitation(iteration, g)

                last = self.mu[g]
                self.mu[g] = last + (1 / self.Np) * np.abs(winner - loser)
                self.sigma[g] = np.sqrt(np.square(self.sigma[g]) + np.square(last) - np.square(self.mu[g]) + (1 / self.Np) * np.abs(np.square(winner) - np.square(loser)))

                # update group best & fmin
                fitness = self.fitness_func(winner)
                if fitness < self.fmin[g]:
                    self.best[g] = winner
                    self.fmin[g] = fitness

                # update global best & fmin
                if self.fmin[g] < self.global_fmin:
                    self.global_best = self.best[g]
                    self.global_fmin = self.fmin[g]

                rand = np.random.rand()
                if rand < 0.5:
                    rand = np.random.rand()
                    if rand < 0.5:
                        # parallel technique 1.

                        rg = int(np.random.uniform(0, self.groups, 1))
                        while rg == g:
                            rg = int(np.random.uniform(0, self.groups, 1))

                        sorted_current = np.sort(self.best[g])
                        sorted_random = np.sort(self.best[rg])
                        mean_current = np.mean(self.best[g])
                        mean_random = np.mean(self.best[rg])
                        slices = (self.dimension + 1) // 2

                        if mean_current < mean_random:
                            part_current = sorted_current[:slices]
                            part_random = sorted_random[:(self.dimension - slices)]
                            new_group = np.append(part_current, part_random)
                            new_fitness = self.fitness_func(new_group)
                            if new_fitness < self.fmin[g]:
                                self.best[g] = new_group
                                self.fmin[g] = new_fitness
                            else:
                                pass
                        else:
                            self.best[g] = self.best[rg]
                            self.fmin[g] = self.fmin[rg]

                        if self.fmin[g] < self.total_best:
                            self.total_best = self.fmin[g]
                    else:
                        # parallel technique 2.

                        sorted_current = np.sort(self.best[g])
                        sorted_global = np.sort(self.global_best)
                        mean_current = np.mean(self.best[g])
                        mean_global = np.mean(self.global_best)
                        slices = (self.dimension + 1) // 2

                        if mean_current < mean_global:
                            part_current = sorted_current[:slices]
                            part_global = sorted_global[:slices]
                            new_group = np.append(part_current, part_global)
                            new_fitness = self.fitness_func(new_group)
                            if new_fitness < self.global_fmin:
                                self.global_best = new_group
                                self.global_fmin = new_fitness
                            else:
                                pass
                        else:
                            self.best[g] = self.global_best
                            self.fmin[g] = self.global_fmin

                        if self.fmin[g] < self.total_best:
                            self.total_best = self.fmin[g]
                        if self.global_fmin < self.total_best:
                            self.total_best = self.global_fmin
                else:
                    # parallel technique 3.

                    rg = int(np.random.uniform(0, self.groups, 1))
                    while rg == g:
                        rg = int(np.random.uniform(0, self.groups, 1))

                    sorted_random = np.sort(self.best[rg])
                    sorted_global = np.sort(self.global_best)
                    mean_random = np.mean(self.best[rg])
                    mean_global = np.mean(self.global_best)
                    slices = (self.dimension + 1) // 2

                    if mean_random < mean_global:
                        part_random = sorted_random[:slices]
                        part_global = sorted_global[:slices]
                        new_group = np.append(part_random, part_global)
                        new_fitness = self.fitness_func(new_group)
                        if new_fitness < self.global_fmin:
                            self.global_best = new_group
                            self.global_fmin = new_fitness
                        else:
                            pass
                    else:
                        self.best[rg] = self.global_best
                        self.fmin[rg] = self.global_fmin

                    if self.fmin[rg] < self.total_best:
                        self.total_best = self.fmin[rg]
                    if self.global_fmin < self.total_best:
                        self.total_best = self.global_fmin

            self.curve[iteration] = self.total_best

        # plot
        plt.xlabel('$Iterations$')
        plt.ylabel('$Best$ $value$')
        plt.plot(np.arange(self.max_iter), self.curve, label='PCGOA')


"""
Test cases
D: dimension
N: population
Iter: max iteration
func: fitness function
"""

D = 100
N = 300
Iter = 100


def func(X):
    return np.mean(X)


lb = np.ones(D) * (-100)
ub = np.ones(D) * 100

goa = GOA(N, D, Iter, lb, ub, func)
cgoa = CGOA(N, D, Iter, lb, ub, func)
pcgoa = PCGOA(N, D, Iter, lb, ub, func)

zone_1 = time.time()
goa.run()
print('GOA: ', time.time() - zone_1, 's')

zone_2 = time.time()
cgoa.run()
print('CGOA: ', time.time() - zone_2, 's')

zone_3 = time.time()
pcgoa.run()
print('PCGOA: ', time.time() - zone_3, 's')

plt.legend()
plt.show()

"""
GOA:  175.8616909980774 s
CGOA:  0.8447632789611816 s
PCGOA:  2.51987886428833 s
"""
