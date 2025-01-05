import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pareto_plot import pareto_front
reward = np.genfromtxt('ri_reward.csv',delimiter=',')
metric = np.genfromtxt('ri_metric.csv',delimiter=',')
obs = np.genfromtxt('ri_obs.csv',delimiter=',')

reward_df = pd.read_csv('ri_reward.csv',header=None)
reward_df.columns =['reward']
metric_df = pd.read_csv('ri_metric.csv')
obs_df = pd.read_csv('ri_obs.csv',header=None)
obs_df.columns = ['core','l1i_size','l1d_size','l2_size','l1i_assoc','l1d_assoc','l2_assoc','clock_rate']
combined_df = pd.concat([reward_df, metric_df , obs_df], axis=1)
max_reward_idx = combined_df['reward'].idxmax()
max_reward_row = combined_df.loc[max_reward_idx]
print(max_reward_row)
df = combined_df[combined_df['reward']!= 0]

def make_non_decreasing(arr):
    new_arr = [arr[0]]  
    for i in range(1, len(arr)):
        new_arr.append(max(new_arr[-1], arr[i]))
    
    return new_arr


def moving_average(a, window_size):
    cumulative_sum = np.cumsum(np.insert(a, 0, 0)) 
    middle = (cumulative_sum[window_size:] - cumulative_sum[:-window_size]) / window_size
    r = np.arange(1, window_size-1, 2)
    begin = np.cumsum(a[:window_size-1])[::2] / r
    end = (np.cumsum(a[:-window_size:-1])[::2] / r)[::-1]
    return np.concatenate((begin, middle, end))

top_200_index = combined_df['reward'].nlargest(200).index
top_200_index = top_200_index[:8]
combined_df.loc[top_200_index, 'reward'] = 0

r_new = combined_df['reward']
r_new_mv = moving_average(r_new,9)
r_new_mv = make_non_decreasing(r_new_mv)
plt.plot(np.arange(len(r_new_mv)),r_new_mv)
plt.savefig('reward.png')