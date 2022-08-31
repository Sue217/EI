# author: Jingbo Su
# Parallel and Compact Gannet Optimization Algorithm

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import gamma
from scipy.stats import truncnorm


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
            self.X[:, d] = np.random.rand(self.groups) * (self.ub[d] - self.lb[d]) + self.lb[d]

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
        mu = np.random.rand(dimension)
        v = np.random.rand(dimension)
        return 0.01 * mu * sigma / ((np.abs(v)) ** (1 / beta))

    def bound_check(self, group):
        self.MX[group] = np.where(self.MX[group] < self.lb, self.lb, self.MX[group])
        self.MX[group] = np.where(self.MX[group] > self.ub, self.ub, self.MX[group])

    def generate(self, group):
        x = np.random.rand(self.dimension)
        a, b = (-1 - self.mu[group]) / self.sigma[group], (1 - self.mu[group]) / self.sigma[group]
        y = truncnorm.cdf(x, a, b, loc=self.mu[group], scale=self.sigma[group])
        y = np.where(np.isnan(y), 1e-9, y)
        x_actual = y * (self.ub - self.lb) / 2 + (self.ub + self.lb) / 2
        return x_actual

    def compete(self, x1, x2):
        if self.fitness_func(x1) < self.fitness_func(x2):
            winner, loser = x1, x2
        else:
            winner, loser = x2, x1
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

        for g in range(self.groups):
            if self.total_best > self.fitness_func(self.X[g]):
                self.total_best = self.fitness_func(self.X[g])
        self.best = self.X

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
                self.mu[g] = last + (1 / self.Np) * (winner - loser)
                self.sigma[g] = np.sqrt(np.abs(np.square(self.sigma[g]) + np.square(last) - np.square(self.mu[g]) + (1 / self.Np) * (np.square(winner) - np.square(loser))))

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
                            part_global = sorted_global[:(self.dimension - slices)]
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
                        part_global = sorted_global[:(self.dimension - slices)]
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
        plt.plot(np.arange(self.max_iter), self.curve, label='PCGOA')