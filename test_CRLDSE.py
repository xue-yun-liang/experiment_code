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


from config import config_global
sys.path.append("./util/")
from space import dimension_discrete, design_space, create_space_gem5
from actor import actor_policyfunction, get_log_prob, get_log_prob_rnn
from mlp import mlp_policyfunction, rnn_policyfunction
from evaluation_gem5 import evaluation_gem5
from config_analyzer import config_self,config_self_new
from timer import timer
from recorder import recorder


debug = False

class RIDSE:
    def __init__(self, iindex):

        self.iindex = iindex

        # random seed setting
        seed = self.iindex * 10000
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        # set config, design space and constraints
        with open('util/config.yaml', 'r') as file:
            config_data = yaml.safe_load(file)

        self.config = config_self_new(config_data)
        self.DSE_action_space = create_space_gem5(config_data)
        self.constraints = self.config.constraints

        # define the hyperparameters
        self.SAMPLE_PERIOD_BOUND = 1
        self.GEMA = 0.999  # RL parameter, discount ratio
        self.ALPHA = 0.001  # RL parameter, learning step rate
        self.THRESHOLD_RATIO = 2  # 0.05
        self.BATCH_SIZE = 1
        self.BASE_LINE = 0
        self.ENTROPY_RATIO = 0
        self.PERIOD_BOUND = 500   # set epochs

        # initial mlp_policyfunction, every action dimension owns a policyfunction
        action_scale_list = list()
        for dimension in self.DSE_action_space.dimension_box:
            action_scale_list.append(int(dimension.get_scale()))
        self.policyfunction = mlp_policyfunction(self.DSE_action_space.get_lenth(), action_scale_list)

        ##initial e_greedy_policy_function
        self.actor = actor_policyfunction()

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

        # set pytorch optimizers
        self.policy_optimizer = torch.optim.Adam(self.policyfunction.parameters(),lr=self.ALPHA)

        #### loss replay buffer, in order to record and reuse high return trace
        self.loss_buffer = list()

        # instantiating the experience buffer(for compute the loss)
        #### data vision related
        self.objectvalue_list = list()
        self.objectvalue_list.append(0)
        self.power_list = list()
        self.period_list = list()
        self.period_list.append(-1)
        self.best_objectvalue = 10000
        self.best_objectvalue_list = list()
        self.best_objectvalue_list.append(self.best_objectvalue)
        self.all_objectvalue = list()
        self.all_objectvalue2 = list()
        self.best_objectvalue2 = 10000
        self.best_objectvalue2_list = list()
        self.best_objectvalue2_list.append(self.best_objectvalue)

        self.action_array = list()
        self.reward_array = list()
        self.desin_point_array = list()
        self.metric_array = list()
        self.noise_std = 0.01


    # 6. Setting up callble loss functions that also provide disgnostics specfic to algo
    def train(self):
        current_status = dict()  # S
        next_status = dict()  # S'

        loss = torch.tensor(0)  # define loss function
        batch_index  = 0

        period_bound = self.SAMPLE_PERIOD_BOUND + self.PERIOD_BOUND
        for period in range(self.PERIOD_BOUND):
            print(f"period:{period}", end="\r")
            self.DSE_action_space.get_status()

            # store log_prob, reward and return
            entropy_list = list()
            log_prob_list = list()
            reward_list = list()
            return_list = list()
            entropy_loss_list = list()

            for step in range(self.DSE_action_space.get_lenth()):
                # get status from S
                current_status = self.DSE_action_space.get_status()

                # use policy function to choose action and acquire log_prob
                entropy, action, log_prob_sampled = self.actor.action_choose(
                    self.policyfunction, self.DSE_action_space, current_status,dimension_index=step
                )
                entropy_list.append(entropy)
                log_prob_list.append(log_prob_sampled)

                # take action and get next state S'
                self.DSE_action_space.sample_one_dimension(step, action)
                next_status = self.DSE_action_space.get_status()
                print("next_status:",next_status)
                objectvalue = 0

                #### in MC method, we can only sample in last step
                # and compute reward R

                if step < (
                    self.DSE_action_space.get_lenth() - 1
                ):  # delay reward, only in last step the reward will be asigned
                    reward = float(0)
                    reward2 = float(0)
                else:
                    
                    metrics = self.evaluation.eval(next_status.values())
                    self.desin_point_array.append(next_status.values())
                    
                    if metrics != None:
                        self.metric_array.append(metrics.values())
                        energy = metrics["latency"]
                        area = metrics["Area"]
                        runtime = metrics["latency"]
                        power = metrics["power"]
                        self.constraints.update({"AREA": area, "POWER": power})

                        reward = 1000 / (runtime * 100000 * self.constraints.get_punishment())
                        objectvalue = runtime
                        objectvalue2 = power
                    else:
                        reward = 0
                        power = 0
                        self.metric_array.append([0,0,0,0])


                    #### recording
                    if (
                        objectvalue < self.best_objectvalue
                        and self.constraints.is_all_meet()
                    ):
                        self.best_objectvalue = objectvalue
                        print(f"best_status:{objectvalue}")
                    if self.constraints.is_all_meet():
                        self.all_objectvalue.append(objectvalue)
                        self.all_objectvalue2.append(objectvalue2)
                    else:
                        self.all_objectvalue.append(10000)
                        self.all_objectvalue2.append(10000)
                    self.best_objectvalue_list.append(self.best_objectvalue)
                    self.period_list.append(period)
                    self.objectvalue_list.append(reward)
                    self.power_list.append(power)

                reward_list.append(reward)

                # assign next_status to current_status
                current_status = next_status

            self.action_array.append(self.DSE_action_space.get_action_list())
            self.reward_array.append(reward)

            # compute and record return
            return_g = 0
            T = len(reward_list)
            for t in range(T):
                return_g = reward_list[T - 1 - t] + self.GEMA * return_g
                return_list.append(torch.tensor(return_g).reshape(1))
            return_list.reverse()


            # compute and record entropy_loss
            entropy_loss = torch.tensor(0)
            T = len(return_list)
            for t in range(T):
                retrun_item = -1 * log_prob_list[t] * (return_list[t] - self.BASE_LINE)
                entropy_item = -1 * self.ENTROPY_RATIO * entropy_list[t]
                entropy_loss = entropy_loss + retrun_item + entropy_item
            entropy_loss = entropy_loss / T

            loss = loss + entropy_loss
            batch_index = batch_index + 1

            # step update policyfunction
            if period % self.BATCH_SIZE == 0:
                loss = loss / self.BATCH_SIZE
                # logger.info(f"entropy_loss:{entropy_loss}")
                self.policy_optimizer.zero_grad()
                loss.backward()
                self.policy_optimizer.step()

                loss = torch.tensor(0)
                batch_index = 0
        # end for-period


    # end def-train
    def save_record(self):
        np.savetxt("record/blackscholes_crldse_reward.csv", self.reward_array, delimiter=',', fmt='%f')
        obs_array = pd.DataFrame(self.desin_point_array)
        obs_array.to_csv("record/blackscholes_crldse_obs.csv",header = None, index = None)
        metric_array = pd.DataFrame(self.metric_array)
        metric_array.to_csv("record/blackscholes_crldse_metric.csv",header = ['latency','Area','energy','power'], index = None)


# running the main loop of the algorithms
def run(iindex):
    print(f"---------------TEST{iindex} START---------------")
    DSE = RIDSE(iindex)
    DSE.train()
    DSE.save_record()
    print(f"---------------TEST{iindex} END---------------")


if __name__ == "__main__":
    TEST_BOUND = 4
    for iindex in range(3, TEST_BOUND):
        run(iindex)
