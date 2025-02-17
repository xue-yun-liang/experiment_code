import torch
import random
import numpy as np
import pdb
import copy
import pandas as pd
import os
import yaml
from multiprocessing import Process, Lock, Manager, Pool
import sys
from config import config_global
sys.path.append("./util/")
from space import dimension_discrete, design_space, create_space_gem5
from actor import actor_e_greedy, status_to_Variable, status_normalize
from replaybuffer import replaybuffer
from mlp import mlp_qfunction
from evaluation_gem5 import evaluation_gem5
from config_analyzer import config_self, config_self_new
from timer import timer
from recorder import recorder

class DQN():
    def __init__(self, iindex):
        self.iindex = iindex
        seed = self.iindex * 10000
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        
        # assign goal, platgorm and constraints
        with open('util/config.yaml', 'r') as file:
            config_data = yaml.safe_load(file)

        self.config = config_self_new(config_data)
        self.DSE_action_space = create_space_gem5(config_data)
        self.constraints = self.config.constraints

        #### step3 assign goal
        # self.goal = self.config.goal
        self.config.config_check()

        #### step1 assign model
        self.goal = self.config.goal
        self.target = self.config.target
        #self.baseline = self.config.baseline
        self.config.config_check()
        self.pid = os.getpid()
        ## initial DSE_action_space

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
        self.evaluation = evaluation_gem5(default_state)
        #hyper parameters
        self.delay_reward = True
        self.PERIOD_BOUND = 1
        self.GEMA = 0.999 #RL parameter, discount ratio, GEMA = 0 means we use reward as return
        self.ALPHA = 0.01 #RL parameter, learning step rate
        self.WAIT_PERIOD = 1 #RL parameter, for fill replaybuffer
        self.BATCH_SIZE = 5 #RL parameter, as its name say
        self.TAU = 0.01
        
        ##initial mlp_qfunction, which input is the vector of status
        '''
        self.qfunction = mlp_qfunction(self.DSE_action_space.get_lenth())
        self.target_qfunction = mlp_qfunction(self.DSE_action_space.get_lenth())
        '''
        self.qfunction = mlp_qfunction(self.DSE_action_space.const_lenth + self.DSE_action_space.dynamic_lenth)
        self.target_qfunction = mlp_qfunction(self.DSE_action_space.const_lenth + self.DSE_action_space.dynamic_lenth)
        self.target_qfunction.load_state_dict(self.qfunction.state_dict())
        ##initial e_greedy_policy_function
        self.actor = actor_e_greedy()
        ##initial replaybuffer
        self.replaybuffer = replaybuffer()
        #initial optimizer
        self.optimizer = torch.optim.Adam(
            self.qfunction.parameters(), 
            lr=self.ALPHA, 
        )
        self.best_objectvalue = 1000
        self.best_objectvalue_list = list()
        self.t = timer()

        self.reward_array = list()
        self.desin_point_array = list()
        self.metric_array = list()

    def train(self):
        current_status = dict() #S
        next_status = dict() #S' 
        loss_batch = torch.tensor(0)
        loss_index = 0
        self.t.start("all")
        for period in range(self.WAIT_PERIOD + self.PERIOD_BOUND):
            #### here may need a initial function for action_space
            ratio = 1 - period/self.PERIOD_BOUND
            self.DSE_action_space.status_reset()
            for step in range(self.DSE_action_space.get_lenth()): 
                #### get status from S
                current_status = self.DSE_action_space.get_status()
                #### use e-greedy algorithm get action A, action is the index of best action
                #### belong to that dimension
                action = self.actor.action_choose_with_no_grad(self.qfunction, self.DSE_action_space, step, ratio)
                #### take action and get next state S'
                self.DSE_action_space.sample_one_dimension(step, action)
                next_status = self.DSE_action_space.get_status()
                if(step < (self.DSE_action_space.get_lenth() - 1)): #delay reward, only in last step the reward will be asigned
                    not_done = 1
                    if(self.delay_reward):
                        reward = float(0)
                    else:
                        all_status = self.DSE_action_space.get_status()
                        self.t.start("eva")
                        metrics = self.evaluation.evaluate(all_status)
                        self.t.end("eva")
                        if(metrics != None):
                            self.metric_array.append(metrics.values())
                            energy = metrics["latency"]
                            area = metrics["Area"]
                            runtime = metrics["latency"]
                            power = metrics["power"]
                            self.constraints.update({"AREA": area,"POWER":power})
                            reward = 1000 / (runtime * 100000 * self.constraints.get_punishment())
                        else:
                            self.metric_array.append([0,0,0,0])
                            reward = 0
                        # print(f"period:{period}, objectvalue:{objectvalue}, reward:{reward}", end = '\r')
                        #### recording
                        # if(period < self.WAIT_PERIOD):
                        #     pass
                        # else:
                        #     if(objectvalue < self.best_objectvalue and self.constraints.is_all_meet()):
                        #         self.best_objectvalue = objectvalue
                        #     self.best_objectvalue_list.append(self.best_objectvalue)
                else:
                    not_done = 0
                    #and compute reward R
                    all_status = self.DSE_action_space.get_status()
                    self.t.start("eva")
                    metrics = self.evaluation.evaluate(all_status)
                    self.t.end("eva")
                    if(metrics != None):
                        self.metric_array.append(metrics.values())
                        energy = metrics["latency"]
                        area = metrics["Area"]
                        runtime = metrics["latency"]
                        power = metrics["power"]
                        self.constraints.update({"AREA": area,"POWER":power})
                        reward = 1000 / (runtime * 100000 * self.constraints.get_punishment())
                    else:
                        self.metric_array.append([0,0,0,0])
                        reward = 0
                    # print(f"period:{period}, objectvalue:{objectvalue}, reward:{reward}", end = '\r')
                    # #### recording
                    # if(period < self.WAIT_PERIOD):
                    #     pass
                    # else:
                    #     if(objectvalue < self.best_objectvalue and self.constraints.is_all_meet()):
                    #         self.best_objectvalue = objectvalue
                    #     self.best_objectvalue_list.append(self.best_objectvalue)
                self.reward_array.append(reward)
                #### push record to replaybuffer
                self.replaybuffer.add(current_status, action, next_status, reward, step, not_done)
                #### assign next_status to current_status
                current_status = next_status
                #################################START TRAIN###########################################
                #### batch train
                if(period < self.WAIT_PERIOD):
                    pass
                else:
                    for index in range(self.BATCH_SIZE):
                        #### randomly sample from replaybuffer
                        sp_current_status, sp_action, sp_next_status, sp_reward, sp_step, sp_not_done = copy.deepcopy(self.replaybuffer.sample())
                        #### compute v(S',w)
                        if(sp_not_done):
                            sp_next_status = status_normalize(sp_next_status, self.DSE_action_space)
                            next_variable = status_to_Variable(sp_next_status)
                            next_cnt = (sp_step + 1) / self.DSE_action_space.get_lenth()
                            next_cnt = torch.tensor(next_cnt).float().view(1)
                            next_variable = torch.cat((next_variable, next_cnt), dim=-1)
                            next_qvalue = self.target_qfunction(next_variable)
                        #### compute return g
                        if(sp_not_done):
                            return_g = torch.tensor(sp_reward).float() + self.GEMA * next_qvalue.detach()
                        else:
                            return_g = torch.tensor(sp_reward).float()
                        #### compute v(S,w)
                        sp_current_status = status_normalize(sp_current_status, self.DSE_action_space)
                        current_variable = status_to_Variable(sp_current_status)
                        current_cnt = sp_step / self.DSE_action_space.get_lenth()
                        current_cnt = torch.tensor(current_cnt).float().view(1)
                        current_variable = torch.cat((current_variable, current_cnt), dim=-1)
                        current_qvalue = self.qfunction(current_variable) 
                        #### train the qfunction
                        loss = torch.nn.functional.mse_loss(current_qvalue, return_g)
                        #### accumalate loss
                        loss_batch = loss_batch + loss
                        loss_index = loss_index + 1
                        #### end for self.BATCH_SIZE
                    #### update network
                    loss_batch = loss_batch / loss_index
                    self.optimizer.zero_grad()
                    loss_batch.backward()
                    self.optimizer.step()
                    loss_batch = torch.tensor(0)
                    loss_index = 0
                    #### gradually update target network
                    for param, target_param in zip(self.qfunction.parameters(), self.target_qfunction.parameters()):
                        target_param.data.copy_(self.TAU * param.data + (1 - self.TAU) * target_param.data)
                #### end of for step
            #### end of for period
        self.t.end("all")
    #### end of def train
    def test(self):
        #### create proxy agent in order to protect status
        proxy_space = copy.deepcopy(self.DSE_action_space)
        proxy_evaluation = copy.deepcopy(self.evaluation)
        for period in range(1):
            for step in range(proxy_space.get_lenth()):
                action = self.actor.best_action_choose(self.qfunction, proxy_space, step)
                self.fstatus = proxy_space.sample_one_dimension(step, action)
            proxy_evaluation.update_parameter(self.fstatus)
            self.fruntime, t_L = proxy_evaluation.runtime()
            self.fpower = proxy_evaluation.power()
            self.fruntime = self.fruntime * 1000
            print(
                "@@@@  TEST  @@@@\n",
                "final_status\n", self.fstatus, 
                "\nfinal_runtime\n", self.fruntime,
                "\nfinal_power\n", self.fpower,
                "\nbest_runtime\n", self.best_runtime_list[-1],
            )
    def qfunction_check(self):
        proxy_space = copy.deepcopy(self.DSE_action_space)
        proxy_evaluation = copy.deepcopy(self.evaluation)
        
        with torch.no_grad():
            for index in range(1000):
                current_status, action, next_status, reward, step, not_done = copy.deepcopy(self.replaybuffer.sample())
                
                if(not_done):
                    next_status = status_normalize(next_status, proxy_space)
                    next_variable = status_to_Variable(next_status)
                    next_cnt = (step+1) / proxy_space.get_lenth()
                    next_cnt = torch.tensor(next_cnt).float().view(1)
                    next_variable = torch.cat((next_variable, next_cnt), dim = -1)
                    next_qvalue = self.target_qfunction(next_variable)
                    return_g = torch.tensor(reward).float() + self.GEMA * next_qvalue.detach()
                else:
                    return_g = torch.tensor(reward).float()
                
                current_status = status_normalize(current_status, proxy_space)
                current_variable = status_to_Variable(current_status)
                current_cnt = step / proxy_space.get_lenth()
                current_cnt = torch.tensor(current_cnt).float().view(1)
                current_variable = torch.cat((current_variable, current_cnt), dim = -1)
                current_qvalue = self.qfunction(current_variable) 
            #     self.worksheet.write(index+1, 0, return_g.item())
            #     self.worksheet.write(index+1, 1, current_qvalue.item())
            # self.workbook.save("record/reward&return/qfunction-TD method.xls")

    def save_record(self):
        np.savetxt("record/erdse_reward.csv", self.reward_array, delimiter=',', fmt='%f')
        np.savetxt("record/erdse_detail.csv", np.stack((self.all_objectvalue,self.all_objectvalue2),axis=1), delimiter=',', fmt='%f')
        obs_array = pd.DataFrame(self.desin_point_array)
        obs_array.to_csv("record/erdse_obs.csv",header=None,index=None)
        metric_array = pd.DataFrame(self.metric_array)
        metric_array.to_csv("record/erdse_metric.csv",header = ['latency','Area','energy','power'], index = None)

# def run(args):
#     iindex, objective_record, timecost_record = args
#     print(f"%%%%TEST{iindex} START%%%%")
#     DSE = DQN(iindex)
#     DSE.train()
#     timecost_list = DSE.t.get_list("all")
#     evacost = DSE.t.get_sum("eva")
#     timecost_list.append(evacost)
#     DSE.best_objectvalue_list.append(iindex)
#     timecost_list.append(iindex)
#     objective_record.append(DSE.best_objectvalue_list)
#     timecost_record.append(timecost_list)
def run(iindex):
    print(f"%%%%TEST{iindex} START%%%%")

    DSE =DQN(iindex)
    print(f"DSE scale:{DSE.DSE_action_space.get_scale()}")
    DSE.train()
    DSE.save_record()

if __name__ == "__main__":
    USE_MULTIPROCESS = False
    TEST_BOUND = 3

    if USE_MULTIPROCESS:
        iindex_list = list()
        for i in range(TEST_BOUND):
            if i < 10 or i >= 20:
                continue
            iindex_list.append(i)

        pool = Pool(1)
        pool.map(run, iindex_list)
        pool.close()
        pool.join()
    else:
        for iindex in range(1, TEST_BOUND):
            run(iindex)
