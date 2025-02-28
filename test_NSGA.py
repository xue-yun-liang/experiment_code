import random
import sys
import os
import yaml
from multiprocessing import Pool

import geatpy as ea
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

sys.path.append("./util/")
from evaluation_gem5 import evaluation_gem5
from config_analyzer import config_self,config_self_new

# global parameters, NIND is the number of population
NIND = 50
with open('util/config.yaml', 'r') as file:
    config_data = yaml.safe_load(file)

config = config_self_new(config_data)
#nnmodel = config.nnmodel
#layer_num = config.layer_num
#has_memory = True
#### step2 assign platform
#constrain
#target = config.target
AREA_THRESHOLD = 60
#### step3 assign goal
config.config_check()
record = 1
my_period =0
best_runtime_now=10000
runtime_list=list()
best_power_now=10000
power_list=list() 

##initial evaluation
default_state = {
    "core": 3,
    "l1i_size": 256,
    "l1d_size": 256,
    "l2_size": 64,
    "l1d_assoc": 8,
    "l1i_assoc": 8,
    "l2_assoc": 8,
    "sys_clock": 2,
}
evaluation = evaluation_gem5(default_state)
constraints = config.constraints

metric_array = list()
reward_array = list()
desin_point_array = list()

class DSE(ea.Problem):
    def __init__(self):
        name = "DSE"
        M = 2               # the dimension of object function
        maxormins = [1,1]   # 1 means to minmize the ob function
        '''
        d1:PE_num d2:PE_size d3:f d4:minibatch d5:bitwidth
        '''
        Dim = 8
        varTypes = [1, 1, 1, 1, 1, 1, 1, 1]     # 1 for discrete value, 0 for continues value
        lb = [1, 1, 1, 6, 1, 1, 1, 20]          # low bound
        ub = [16, 12, 12, 16, 4, 4, 4, 40]      # up bound
        lbin = [1] * Dim                        # 1 means include the low bound, 0 means do not include the low bound
        ubin = [1] * Dim
        ea.Problem.__init__(self, name, M, maxormins, Dim, varTypes, lb, ub, lbin, ubin)
        output_dir = "./data"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # 定义输出文件路径
        output_file = f"{output_dir}/{config.benchmark}_{config.target}_RANDOM_DATA.csv"

        # 如果文件不存在，先创建并写入表头
        if not os.path.exists(output_file):
            with open(output_file, 'w') as f:
                f.write("reward,core,l1i_size,l1d_size,l2_size,l1i_assoc,l1d_assoc,l2_assoc,clock_rate,latency,Area,energy,power\n")


        self.save_path = output_file
        
    def aimFunc(self, pop):
        global my_period,best_runtime_now,best_power_now,power_list,runtime_list
        vars = pop.Phen # Phen is the variables matrix
        vec_core = vars[:, [0]]
        vec_l1i_size = vars[:, [1]]
        vec_l1d_size = vars[:, [2]]
        vec_l2_size = vars[:, [3]]
        vec_l1d_assoc = vars[:, [4]]
        vec_l1i_assoc = vars[:, [5]]
        vec_l2_assoc = vars[:, [6]]
        vec_sys_clock = vars[:, [7]]
        vec_runtime = np.zeros(NIND).reshape(-1,1)
        vec_energy = np.zeros(NIND).reshape(-1,1)
        vec_power = np.zeros(NIND).reshape(-1,1)
        vec_area = np.zeros(NIND).reshape(-1, 1)
        for index in range(NIND):
            print("+++++++++++++++++++" + str(my_period + 1) + "++++++++++++++++++++++++++")
            my_period = my_period + 1
            status = dict()
            status['core'] = int(vec_core[index])
            status['l1i_size'] = 2**int(vec_l1i_size[index])
            status['l1d_size'] = 2**int(vec_l1d_size[index])
            status['l2_size'] = 2**int(vec_l2_size[index])
            status['l1d_assoc'] = 2**int(vec_l1d_assoc[index])
            status['l1i_assoc'] = 2**int(vec_l1i_assoc[index])
            status['l2_assoc'] = 2**int(vec_l2_assoc[index])
            status['sys_clock'] = int(vec_sys_clock[index])/10
            desin_point_array.append(status.values())
            metrics = evaluation.eval(status.values())
            if (metrics != None):
                energy = metrics['energy']
                area = metrics['Area']
                runtime = metrics['latency']
                power = metrics['power']
                metric_array.append(metrics.values())
                constraints.update({"AREA": area, "POWER": power})
                reward = 1000 / (runtime * 100000 * constraints.get_punishment())
                reward_array.append(reward)
            else:
                runtime = 10000
                power = 10000
                energy = 10000
                area =10000
                metric_array.append([10000, 10000, 10000, 10000])
                reward_array.append(-1)
                metrics = {"energy": 0, "Area": 0, "latency": 0, "power": 0}
            with open(self.save_path, 'a') as f:
                f.write(f"{reward},{status['core']},{status['l1i_size']},{status['l1d_size']},{status['l2_size']},{status['l1i_assoc']},{status['l1d_assoc']},{status['l2_assoc']},{status['sys_clock']},{metrics['latency']},{metrics['Area']},{metrics['energy']},{metrics['power']}\n")

            if  area < AREA_THRESHOLD :
                best_runtime_now = runtime
                best_power_now = power
            else:
                best_runtime_now = 10000
                best_power_now = 10000
            runtime_list.append(best_runtime_now)
            power_list.append(best_power_now)
            vec_runtime[index] = runtime
            vec_energy[index] = energy
            vec_power[index] = power
            vec_area[index] = area - AREA_THRESHOLD
        pop.ObjV = np.hstack([vec_runtime,vec_power])
        pop.CV = np.hstack([vec_area])

    def calReferObjV(self):
        referenceObjV = np.array([[50,0],[0,0]])
        return referenceObjV
    
def run(iindex):
    global reward_array, metric_array, desin_point_array
    print(f"%%%%%%%%%%%%%%%TEST{iindex} START%%%%%%%%%%%%%")
    global my_period, best_runtime_now, best_power_now,power_list, runtime_list

    seed = iindex * 10000
    atype = int(iindex / 10)
    np.random.seed(seed)
    random.seed(seed)
    problem = DSE()
    Encoding = "RI" 
    Field = ea.crtfld(Encoding, problem.varTypes, problem.ranges, problem.borders)
    population = ea.Population(Encoding, Field, NIND)
    myalgorithm = ea.moea_NSGA2_templet(problem, population)
    myalgorithm.MAXGEN = 10
    #myalgorithm.mutOper.F = 0.5
    #myalgorithm.recOper.XOVR = 0.7
    myalgorithm.drawing = 1
    print("________________algorithm run_________________")
    referenceObjV = np.array([50])
    NDSet = myalgorithm.run()
    print(f"**************TEST{iindex} END***********")
    reward_array = pd.DataFrame(reward_array,columns=["reward"])
    obs_array = pd.DataFrame(desin_point_array,columns=["core","l1i_size","l1d_size","l2_size","l1i_assoc","l1d_assoc","l2_assoc","clock_rate"])
    metric_array = pd.DataFrame(metric_array,columns=['latency','Area','energy','power'])
    result_df = pd.concat([reward_array,obs_array,metric_array], axis=1)
    result_df.to_csv(f"./data/{config.benchmark}_{config.target}_nsga2_f.csv")

if __name__ == '__main__':
    USE_MULTIPROCESS = False
    TEST_BOUND = 1
    if(USE_MULTIPROCESS):
        iindex_list = list()
        for i in range(TEST_BOUND):
            iindex_list.append(i)
        pool = Pool(3)
        pool.map(run, iindex_list)
        pool.close()
        pool.join()
    else:
        for iindex in range(TEST_BOUND):
            run(iindex)