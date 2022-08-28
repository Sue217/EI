# author: Jingbo Su
# Gannet Optimization Algorithm

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import gamma


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
        