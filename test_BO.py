from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from multiprocessing import Process, Lock, Manager, Pool
import random
import pdb
import copy
from multiprocessing import Pool

import gym
import numpy as np
import pandas as pd
import torch
import yaml
import sys
import os


from config import config_global
sys.path.append("./util/")
from space import dimension_discrete, design_space, create_space_gem5
from actor import actor_policyfunction, get_log_prob, get_log_prob_rnn, self_attention, seq_len_encoder
from mlp import mlp_policyfunction, rnn_policyfunction, mlp
from evaluation_gem5 import evaluation_gem5
from config_analyzer import config_self,config_self_new
from timer import timer
from recorder import recorder

def object_function(parameters):
    design_space = parameters["design_space"]
    config = parameters["config"]
    evaluation = parameters["evaluation"]
    record = parameters["record"]
    t = parameters["t"]
    algo = 'bo'

    constraints = config.constraints

    output_dir = "./data"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    output_file = f"{output_dir}/{config.benchmark}_{config.target}_{algo}.csv"

    names = record.names
    action_list = list()
    for name in names:
        action_list.append(parameters[name])
    print("action_list:",action_list)
    status = design_space.status_set(action_list)
    t.start("eva")
    metrics = evaluation.eval(status.values())
    t.end("eva")


    metric_array = []
    if metrics != None:
        metric_array.append(metrics.values())
        energy = metrics["latency"]
        area = metrics["Area"]
        runtime = metrics["latency"]
        power = metrics["power"]
        constraints.update({"AREA": area, "POWER": power})

        reward = 1000 / (runtime * 100000 * constraints.get_punishment())
        objectvalue = runtime
        objectvalue2 = power
    else:
        reward = 0
        power = 0
        metrics = {"latency":0, "power":0, "Area":0,"energy":0}
    with open(output_file, 'a') as f:
        f.write(f"{reward},{status['core']},{status['l1i_size']},{status['l1d_size']},{status['l2_size']},{status['l1i_assoc']},{status['l1d_assoc']},{status['l2_assoc']},{status['sys_clock']},{metrics['latency']},{metrics['Area']},{metrics['energy']},{metrics['power']}\n")

    # if(not record.objectvalue_list):
    #     record.objectvalue_list.append(objectvalue)
    # else:
    #     pass
    # best_objectvalue = record.objectvalue_list[-1]

    # if(objectvalue < best_objectvalue and constraints.is_all_meet()):
    #     best_objectvalue = objectvalue
    #     print(f"objectvalue:{objectvalue}, reward:{reward}, metrics:{metrics}", end = '\r')
    # record.objectvalue_list.append(best_objectvalue)
    # record.multiobjecvalue_list.append([metrics["latency"], metrics["energy"]])
    
    return {"loss": reward, "status": STATUS_OK}

def run(args):
    iindex, objective_record, timecost_record, multiobjective_record = args
    print(f"%%%%%%%%%%%%%%%TEST{iindex} START%%%%%%%%%%%%%")
    seed = iindex * 10000
    np.random.seed(seed)
    random.seed(seed)
    with open('util/config.yaml', 'r') as file:
        config_data = yaml.safe_load(file)

    config = config_self_new(config_data)
    design_space = create_space_gem5(config_data)
    constraints = config.constraints

    target = config.target
    period = 500
    config.config_check()
    pid = os.getpid()

    design_space = create_space_gem5(config_data)
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

    lbs = [0] * design_space.get_lenth()
    ubs = design_space.get_dimension_scale_list(has_upbound = True)
    names = list(design_space.get_status().keys())
    class record():
        def __init__(self):
            self.objectvalue_list = list()
            self.names = list(design_space.get_status().keys())
            self.multiobjecvalue_list = list()
    record = record()
    t = timer()

    space = dict()
    for lb, ub, name in zip(lbs, ubs, names):
        space[name] = hp.choice(name, range(lb, ub))

    space["design_space"] = hp.choice("design_space", [design_space])
    space["config"] = hp.choice("config", [config])
    space["evaluation"] = hp.choice("evaluation", [evaluation])
    space["record"] = hp.choice("record", [record])
    space["t"] = hp.choice("t", [t])

    trials_list = list()
    trials = Trials()
    trials_list.append(trials)
    t.start("all")
    result = fmin(fn = object_function, space = space, algo = tpe.suggest, max_evals = period, trials = trials)
    t.end("all")
    timecost_list = t.get_list("all")
    evacost = t.get_sum("eva")
    timecost_list.append(evacost)

    record.objectvalue_list.append(iindex)
    timecost_list.append(iindex)
    record.multiobjecvalue_list.append([iindex, iindex])
    objective_record.append(record.objectvalue_list)
    timecost_record.append(timecost_list)
    multiobjective_record.append(record.multiobjecvalue_list)
    print(f"%%%%TEST{iindex} END%%%%")

if __name__ == '__main__':
    algoname = "BO_MOO"
    use_multiprocess = True
    global_config = config_global()
    TEST_BOUND = global_config.TEST_BOUND
    #PROCESS_NUM = global_config.PROCESS_NUM
    PROCESS_NUM = 5
    SCEN_TYPE = global_config.SCEN_TYPE
    SCEN_NUM = global_config.SCEN_NUM
    PASS = global_config.PASS

    args_list = list()
    objective_record = Manager().list()
    timecost_record = Manager().list()
    multiobjective_record = Manager().list()

    if(use_multiprocess):
        args_list = list()
        for iindex in range(TEST_BOUND):
            if(iindex in PASS): continue
            args_list.append((iindex, objective_record, timecost_record, multiobjective_record))
        pool = Pool(PROCESS_NUM)
        pool.map(run, args_list)
        pool.close()
        pool.join()
    else:
        for iindex in range(TEST_BOUND):
            if(iindex in PASS): continue
            run((iindex, objective_record, timecost_record))
