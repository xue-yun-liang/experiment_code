#encoding: utf-8
import numpy as np
import xlwt
import yaml
import sys
import pandas as pd
sys.path.append("./util/")
from mopso import Mopso
from config_analyzer import config_self,config_self_new

with open('util/config.yaml', 'r') as file:
    config_data = yaml.safe_load(file)
config = config_self_new(config_data)
#nnmodel = config.nnmodel
#layer_num = config.layer_num
has_memory = True
#### step2 assign platform
#constrain
# target = config.target
# DSP_THRESHOLD = config.DSP_THRESHOLD
# POWER_THRESHOLD = config.POWER_THRESHOLD
# BW_THRESHOLD = config.BW_THRESHOLD
# BRAM_THRESHOLD = config.BRAM_THRESHOLD

reward_array = list()
desin_point_array = list()
metric_array = list()


def main():
    w = 0.8         #惯性因子
    c1 = 0.1        #局部速度因子
    c2 = 0.1        #全局速度因子
    # particals = 50  #粒子群的数量
    particals = 1  #粒子群的数量
    # cycle_ = 10     #迭代次数
    cycle_ = 1     #迭代次数
    mesh_div = 10   #网格等分数量
    thresh = 300    #外部存档阀值
    min_ = np.array([1, 1, 1, 6, 1, 1, 1, 20])          #粒子坐标的最小值
    max_ = np.array([16, 12, 12, 16, 4, 4, 4, 40])      #粒子坐标的最大值
    mopso_ = Mopso(particals,w,c1,c2,max_,min_,thresh,mesh_div)         #粒子群实例化
    pareto_in, pareto_fitness = mopso_.done(cycle_)                      #经过cycle_轮迭代后，pareto边界粒子
    pareto_fitness.tolist()
    np.savetxt("./data/mopso_pareto_in.txt",pareto_in)#保存pareto边界粒子的坐标
    # np.savetxt("./img_txt/pareto_fitness.txt",pareto_fitness) #打印pareto边界粒子的适应值
    reward_array = pd.DataFrame(mopso_.reward_array,columns=["reward"])
    obs_array = pd.DataFrame(mopso_.design_point_array,columns=["core","l1i_size","l1d_size","l2_size","l1i_assoc","l1d_assoc","l2_assoc","clock_rate"])
    metric_array = pd.DataFrame(mopso_.metric_array,columns=['latency','Area','power'])
    result_df = pd.concat([reward_array,obs_array,metric_array], axis=1)
    result_df.to_csv("./data/blackscholes_test_mopso.csv")
    print ("\n,迭代结束")
 
if __name__ == "__main__":
    main()