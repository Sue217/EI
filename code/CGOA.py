# author: Jingbo Su
# Compact Gannet Optimization Algorithm

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import gamma
from scipy.stats import truncnorm


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
        