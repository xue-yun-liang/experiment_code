#encoding: utf-8
import numpy as np
from fitness_funs import *
import init
import update
import pandas as pd

class Mopso:
    def __init__(self,particals,w,c1,c2,max_,min_,thresh,mesh_div=10):
        self.w,self.c1,self.c2 = w,c1,c2
        self.mesh_div = mesh_div
        self.particals = particals
        self.thresh = thresh
        self.max_ = max_
        self.min_ = min_
        self.max_v = (max_-min_)*0.05  #速度上限
        self.min_v = (max_-min_)*0.05*(-1) #速度下限
        #self.plot_ = plot.Plot_pareto()
        self.bestruntime = 10000
        self.bestpower = 10000
        self.runtime_list = list()
        self.power_list = list()
        self.area_list = list()
        self.design_point_array = list()
        self.metric_array = list()
        self.reward_array = list()
    def evaluation_fitness(self): 
        #计算适应度值ֵ
        fitness_curr = []
        for i in range((self.in_).shape[0]):
            print(f"+++++++++++++++++++++++++++++number{i}++++++++++++++++++++++")
            fitness_cheack,singnal,status,metric,reward = fitness_(self.in_[i])
            self.metric_array.append(metric)
            self.reward_array.append(reward)
            self.design_point_array.append(status)
            print(singnal)
            fitness_curr.append(fitness_cheack)
            if singnal:
                self.bestpower = fitness_cheack[0]
                self.bestruntime = fitness_cheack[1]
                self.bestarea = fitness_cheack[2]

            else:
                self.bestpower = 10000
                self.bestruntime = 10000
                self.bestarea = 10000
            self.runtime_list.append(self.bestruntime)
            self.power_list.append(self.bestpower)
            self.area_list.append(self.bestarea)
        self.fitness_ = np.array(fitness_curr)

    def initialize(self):
        self.in_ = init.init_designparams(self.particals,self.min_,self.max_)   #初始化粒子位置
        self.v_ = init.init_v(self.particals,self.min_v,self.max_v)             #初始化粒子速度
        self.evaluation_fitness()                           #计算适应度ֵ
        self.in_p,self.fitness_p = init.init_pbest(self.in_,self.fitness_)      #初始化个体最优
        self.archive_in,self.archive_fitness = init.init_archive(self.in_,self.fitness_)    #初始化外部存档
        self.in_g,self.fitness_g = init.init_gbest(self.archive_in,self.archive_fitness,self.mesh_div,self.min_,self.max_,self.particals)   #初始化全局最优

    def update_(self):
        #更新粒子速度、位置、适应度、个体最优、外部存档、全局最优
        self.v_ = update.update_v(self.v_,self.min_v,self.max_v,self.in_,self.in_p,self.in_g,self.w,self.c1,self.c2)
        self.in_ = update.update_in(self.in_,self.v_,self.min_,self.max_)
        self.evaluation_fitness()
        self.in_p,self.fitness_p = update.update_pbest(self.in_,self.fitness_,self.in_p,self.fitness_p)
        self.archive_in,self.archive_fitness = update.update_archive(self.in_,self.fitness_,self.archive_in,self.archive_fitness,self.thresh,self.mesh_div,self.min_,self.max_,self.particals)
        self.in_g,self.fitness_g = update.update_gbest(self.archive_in,self.archive_fitness,self.mesh_div,self.min_,self.max_,self.particals)

    def done(self,cycle_):
        self.initialize()
        #self.plot_.show(self.in_,self.fitness_,self.archive_in,self.archive_fitness,-1)
        for i in range(cycle_):
            print(f"+++++++++++++++++++++++++++++cycle{i}++++++++++++++++++++++")
            self.update_()
            #self.plot_.show(self.in_,self.fitness_,self.archive_in,self.archive_fitness,i)
        return self.archive_in,self.archive_fitness