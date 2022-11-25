
# -*- coding: utf-8 -*-
"""
Created on Tue May 17 17:06:19 2016

@author: Hossam Faris
"""


class solution:
    def __init__(self):
        self.best = 0
        self.bestIndividual = []
        self.convergence = []
        self.optimizer = ""
        self.objfname = ""
        self.startTime = 0
        self.endTime = 0
        self.executionTime = 0
        self.lb = 100
        self.ub = -100
        self.dim = 10
        self.popnum = 30
        self.maxiers = 0
"""
Created on  20/11/2022

@author: Doaa Abdelkareem
"""

import random
import numpy
import math
from scipy import stats
#from solution import solution
import time


def DO(objf, lb, ub, dim, SearchAgents_no, Max_iter):
    if not isinstance(lb, list):
        lb = [lb] * dim
    if not isinstance(ub, list):
        ub = [ub] * dim

    # Best_score = float("inf")  # change this to -inf for maximization problems
    Convergence_curve = numpy.zeros(Max_iter)
    solution=solution()
    # Initialize the positions of search agents
    dandelions_fitness = numpy.zeros(SearchAgents_no)
    dandelions = numpy.zeros((SearchAgents_no, dim))
    for i in range(dim):
        dandelions[:, i] = numpy.random.uniform(0, 1, SearchAgents_no) * (ub[i] - lb[i]) + lb[i]
    s = solution()

    print('DO is optimizing  "' + objf.__name__ + '"')

    timerStart = time.time()
    s.startTime = time.strftime("%Y-%m-%d-%H-%M-%S")

    for i in range(0, N):
        # evaluate moths
        dandelions_fitness[i] = objf(dandelions[i, :])

    sorted_dandelions_fitness = numpy.sort(dandelions_fitness)
    I = numpy.argsort(dandelions_fitness)

    Sorted_dandelions = numpy.copy(dandelions[I, :])

    Best_dandelions = numpy.copy(Sorted_dandelions[0, :])
    Best_score = sorted_dandelions_fitness[0]
    Convergence_curve[1] = Best_score
    t = 2

    while t < Max_iter + 1:
        # Rising stage ***********************
        beta = numpy.random.randn(SearchAgents_no, dim)
        alpha = random.random() * ((1 / Max_iter ** 2) * t ** 2 - 2 / Max_iter * t + 1)  # eq.(8) in this paper
        a = -1 / (Max_iter ** 2 - 2 * Max_iter + 1)
        b = -2 * a
        c = 1 - a - b
        k = 1 - random.random() * (c + a * t ^ 2 + b * t)  # eq.(11) in this paper
        if numpy.random.randn() < 1.5:
            for i in range(0, SearchAgents_no):
                lamb = abs(numpy.random.randn(1, dim))
                theta = (2 * random.random() - 1) * math.pi
                row = 1 / math.exp(theta)
                vx = row * math.cos(theta)
                vy = row * math.sin(theta)
                NEW = numpy.multiply(random.random(1, dim), (ub - lb)) + lb
                y=numpy.multiply(alpha, vx)
                x=numpy.multiply(y, vx)
                z=stats.lognorm(lamb, 0, 1)
                w=numpy.multiply(x, z)
                g=NEW(1,arange()) - dandelions(i,arange())
                h=numpy.multiply(w, g)
                dandelions_1[i,arange()]=dandelions(i,arange()) + h
                    
        else:
            for i in range(0, SearchAgents_no):
                dandelions_1[i,:]= numpy.multiply(dandelions(i,arange()), k)
       
    
                                                                        
                                                        
        dandelions = dandelions_1.copy()

                    # Decline stage *****************************
        dandelions_mean = sum(dandelions, 1) / SearchAgents_no  # eq.(14) in this paper
        for i in range(0, SearchAgents_no):
            for j in range(dim):
                dandelions_2[i, j] = dandelions(i, j) - beta(i, j) * alpha *(dandelions_mean(1, j) - beta(i, j) * alpha * dandelions(i, j))
                            

        dandelions = dandelions_2.copy()

                    # Landing stage **************************
        Step_length = levy(SearchAgents_no, dim, 1.5)
        Elite = numpy.tile(Best_position, SearchAgents_no, 1)
        for i in range(0, SearchAgents_no):
            for j in range(dim):
                dandelions_3[i, j] = Elite(i, j) + Step_length(i, j) * alpha * (Elite(i, j) - dandelions(i, j) * (2 * t / Max_iter))
                                        

        dandelions = dandelions_3.copy()

        for i in range(0, SearchAgents_no):
                        # evaluate fitness
            dandelions_fitness[i] = objf(dandelions[i, :])

        sorted_dandelions_fitness = numpy.sort(dandelions_fitness)
        I = numpy.argsort(dandelions_fitness)

        Sorted_dandelions = numpy.copy(dandelions[I, :])

        dandelions = numpy.copy(Sorted_dandelions[0, :])
        SortfitbestN = sorted_dandelions_fitness[0]

        if SortfitbestN(1) < Best_fitness:
            Best_dandelions = numpy.copy(dandelions[i, :])
            Best_score = SortfitbestN[i]

        convergence_curve[t] = Best_score

        if t % 1 == 0:
                        print(
                            ["At iteration " + str(t) + " the best fitness is " + str(Best_score)]
                        )
        t = t + 1

    timerEnd = time.time()
    s.endTime = time.strftime("%Y-%m-%d-%H-%M-%S")
    s.executionTime = timerEnd - timerStart
    s.convergence = convergence_curve
    s.optimizer = "DO"
    s.objfname = objf.__name__
    s.best = Best_score
    s.bestIndividual = Best_dandelions

    return s


def levy(n, m, beta):

                # beta is set to 1.5 in this paper
                num = math.gamma(1 + beta) * math.sin(pi * beta / 2)
                # DO.m:116
                den = math.gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2)
                # DO.m:117
                sigma_u = (num / den) ** (1 / beta)
                # DO.m:118
                u = numpy.random.normal(0, sigma_u, n, m)
                # DO.m:119
                v = numpy.random.normal(0, 1, n, m)
                # DO.m:120
                z = u / (abs(v) ** (1 / beta))
                # DO.m:121
                return z



