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


sys.path.append("./util/")
from space import dimension_discrete, design_space, create_space_gem5
from actor import actor_policyfunction, get_log_prob, get_log_prob_rnn
from mlp import mlp_policyfunction, rnn_policyfunction
from evaluation_gem5 import evaluation_gem5
from config_analyzer import config_self,config_self_new
from timer import timer
from recorder import recorder

debug = False


class ERDSE:
    def __init__(self, iindex):

        self.iindex = iindex

        seed = self.iindex * 10000
        # atype = int(self.iindex / 10)
        atype = 4

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

        # define the hyperparameters
        self.PERIOD_BOUND = 1       # set epochs
        self.SAMPLE_PERIOD_BOUND = 1
        self.GEMA = 0.999  # RL parameter, discount ratio
        self.ALPHA = 0.001  # RL parameter, learning step rate
        self.THRESHOLD_RATIO = 2
        self.BATCH_SIZE = 1
        self.BASE_LINE = 0
        self.ENTROPY_RATIO = 0.1

        # initial mlp_policyfunction, every action dimension owns a policyfunction
        # TODO:share the weight of first two layer among policyfunction
        action_scale_list = list()
        for dimension in self.DSE_action_space.dimension_box:
            action_scale_list.append(int(dimension.get_scale()))
        self.policyfunction = mlp_policyfunction(
            self.DSE_action_space.get_lenth(), action_scale_list
        )

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
        

        ##initial optimizer
        self.policy_optimizer = torch.optim.Adam(
            self.policyfunction.parameters(),
            lr=self.ALPHA,
        )

        #### replay buffer, in order to record and reuse high return trace
        #### buffer is consist of trace list
        self.return_buffer = list()
        self.status_buffer = list()
        self.action_buffer = list()
        self.step_buffer = list()
        self.probs_noise_buffer = list()

        #### data vision related
        self.objectvalue_list = list()
        self.objectvalue_list.append(0)
        self.objectvalue_list2 = list()
        self.objectvalue_list2.append(0)

        self.objectvalue2_list = list()
        self.objectvalue2_list.append(0)

        self.period_list = list()
        self.period_list.append(-1)
        self.best_objectvalue = 10000
        self.best_objectvalue_list = list()
        self.best_objectvalue_list.append(self.best_objectvalue)

        self.all_objectvalue = list()
        self.all_objectvalue2 = list()
        self.all_objectvalue3 = list()

        self.best_objectvalue2 = 10000
        self.best_objectvalue2_list = list()
        self.best_objectvalue2_list.append(self.best_objectvalue)

        self.power_list = list()

        self.action_array = list()
        self.reward_array = list()
        self.desin_point_array = list()
        self.metric_array = list()

    def train(self):
        current_status = dict()  # S
        next_status = dict()  # S'
        reward_log_name = (
            "record/objectvalue/"
            + "reward_"
            + "ERDSE"
            + "_"
            + str(self.iindex)
            + ".txt"
        )
        reward_log = open(reward_log_name, "w")

        period_bound = self.SAMPLE_PERIOD_BOUND + self.PERIOD_BOUND
        for period in range(period_bound):
            print(f"period:{period}", end="\r")
            print(period)
            # here may need a initial function for action_space
            self.DSE_action_space.status_reset()

            # store status, log_prob, reward and return
            status_list = list()
            step_list = list()
            action_list = list()
            reward_list = list()
            return_list = list()
            probs_noise_list = list()

            batch_index = 0

            for step in range(self.DSE_action_space.get_lenth()):
                step_list.append(step)
                # get status from S
                current_status = self.DSE_action_space.get_status()
                status_list.append(current_status)

                # use policy function to choose action and acquire log_prob
                action= self.actor.action_choose_with_no_grad(
                    self.policyfunction, self.DSE_action_space, current_status,step
                )
                action_list.append(action)
                # probs_noise_list.append(probs_noise)

                # take action and get next state S'
                self.DSE_action_space.sample_one_dimension(step, action)
                next_status = self.DSE_action_space.get_status()

                #### in MC method, we can only sample in last step
                #### and compute reward R

                # TODO:design a good reward function
                if step < (
                    self.DSE_action_space.get_lenth() - 1
                ):  # delay reward, only in last step the reward will be asigned
                    reward = float(0)
                    reward2 = float(0)
                else:

                    metrics = self.evaluation.eval(next_status.values())
                    self.desin_point_array.append(next_status.values())
                    self.metric_array.append(metrics)
                    if metrics != None:

                        energy = metrics["latency"]
                        area = metrics["Area"]
                        runtime = metrics["latency"]
                        power = metrics["power"]
                        self.constraints.update({"AREA": area,"POWER":power})

                        reward = 1000 / (
                            runtime * 100000 * self.constraints.get_punishment()
                        )
                        objectvalue = runtime
                        objectvalue2 = power
                        objectvalue3 = energy
                        print(reward, self.constraints.get_punishment())
                    else:
                        reward = 0
                    #### recording
                    if period < self.SAMPLE_PERIOD_BOUND:
                        pass
                    else:
                        if (
                            objectvalue < self.best_objectvalue
                            and self.constraints.is_all_meet()
                        ):
                            self.best_objectvalue = objectvalue
                            # logger.info(f"best_status:{objectvalue, objectvalue2, power, DSP, BW, BRAM}")
                        if self.constraints.is_all_meet():
                            self.all_objectvalue.append(objectvalue)
                            self.all_objectvalue2.append(objectvalue2)
                            self.all_objectvalue3.append(objectvalue3)
                        self.best_objectvalue_list.append(self.best_objectvalue)
                        self.best_objectvalue2_list.append(self.best_objectvalue2)
                        self.period_list.append(period - self.SAMPLE_PERIOD_BOUND)
                        self.objectvalue_list.append(reward)
                        self.objectvalue_list2.append(reward2)
                        self.power_list.append(power)
                        print(f"{period}\t{reward}", end="\n", file=reward_log)

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


            #### record trace into buffer
            self.use_max_value_expectation = True
            if self.use_max_value_expectation:
                if len(self.return_buffer) < 1:
                    self.return_buffer.append(return_list)
                    self.status_buffer.append(status_list)
                    self.action_buffer.append(action_list)
                    self.step_buffer.append(step_list)
                    self.probs_noise_buffer.append(probs_noise_list)
                else:
                    min_index = np.argmin(self.return_buffer, axis=0)
                    min_index = min_index[0]
                    min_return = self.return_buffer[min_index][0]
                    if return_list[0] > min_return:
                        self.return_buffer[min_index] = return_list
                        self.status_buffer[min_index] = status_list
                        self.action_buffer[min_index] = action_list
                        self.step_buffer[min_index] = step_list
                        self.probs_noise_buffer[min_index] = probs_noise_list
                    else:
                        pass
            else:  #### use high value trace replay
                if return_list[0] > 8:
                    self.return_buffer.append(return_list)
                    self.status_buffer.append(status_list)
                    self.action_buffer.append(action_list)
                    self.step_buffer.append(step_list)
                    self.probs_noise_buffer.append(probs_noise_list)

            if period < self.SAMPLE_PERIOD_BOUND:
                pass
            elif len(self.return_buffer) > 0:
                #### compute loss and update actor network
                loss = torch.tensor(0)
                for _ in range(self.BATCH_SIZE):
                    #### random sample trace from replay buffer
                    sample_index = random.randint(0, len(self.return_buffer) - 1)
                    s_return_list = self.return_buffer[sample_index]
                    s_status_list = self.status_buffer[sample_index]
                    s_action_list = self.action_buffer[sample_index]
                    s_step_list = self.step_buffer[sample_index]
                    s_probs_noise_list = self.probs_noise_buffer[sample_index]

                    #### compute log_prob and entropy
                    T = len(s_return_list)
                    sample_loss = torch.tensor(0)
                    pi_noise = torch.tensor(1)
                    for t in range(T):
                        s_entropy, s_log_prob = get_log_prob(
                            self.policyfunction,
                            self.DSE_action_space,
                            s_status_list[t],
                            s_action_list[t],
                            s_step_list[t],
                        )
                        retrun_item = (
                            -1 * s_log_prob * (s_return_list[t] - self.BASE_LINE)
                        )
                        entropy_item = -1 * self.ENTROPY_RATIO * s_entropy
                        sample_loss = sample_loss + retrun_item + entropy_item
                        # pi_noise = pi_noise * s_probs_noise_list[t].detach()
                    #### accumulate loss
                    sample_loss = sample_loss / T
                    loss = loss + sample_loss
                loss = loss / self.BATCH_SIZE
                # logger.info(loss)

                # logger.info(f"loss:{loss}")
                self.policy_optimizer.zero_grad()
                loss.backward()
                self.policy_optimizer.step()
            else:
                print("no avaiable sample")
        # end for-period

    # end def-train

    def save_record(self):
        np.savetxt("record/erdse_reward.csv", self.reward_array, delimiter=',', fmt='%f')
        np.savetxt("record/erdse_detail.csv", np.stack((self.all_objectvalue,self.all_objectvalue2),axis=1), delimiter=',', fmt='%f')
        obs_array = pd.DataFrame(self.desin_point_array)
        obs_array.to_csv("record/erdse_obs.csv",header=None,index=None)
        metric_array = pd.DataFrame(self.metric_array)
        metric_array.to_csv("record/erdse_metric.csv",header = None, index = None)



def run(iindex):
    print(f"%%%%TEST{iindex} START%%%%")

    DSE =ERDSE(iindex)
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
