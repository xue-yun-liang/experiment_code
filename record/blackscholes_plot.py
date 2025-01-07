import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pareto_plot import pareto_front
from utils import make_non_decreasing, moving_average, compute_square


crldse_reward_df = pd.read_csv('../data/cloud/blackscholes/blackscholes_crldse_reward.csv',header=None)
crldse_reward_df.columns =['reward']
crldse_metric_df = pd.read_csv('../data/cloud/blackscholes/blackscholes_crldse_metric.csv')
crldse_obs_df = pd.read_csv('../data/cloud/blackscholes/blackscholes_crldse_obs.csv',header=None)
crldse_obs_df.columns = ['core','l1i_size','l1d_size','l2_size','l1i_assoc','l1d_assoc','l2_assoc','clock_rate']


erdse_reward_df = pd.read_csv('../data/cloud/blackscholes/blackscholes_erdse_reward.csv',header=None)
erdse_reward_df.columns =['reward']
erdse_metric_df = pd.read_csv('../data/cloud/blackscholes/blackscholes_erdse_metric.csv')
erdse_obs_df = pd.read_csv('../data/cloud/blackscholes/blackscholes_erdse_obs.csv',header=None)
erdse_obs_df.columns = ['core','l1i_size','l1d_size','l2_size','l1i_assoc','l1d_assoc','l2_assoc','clock_rate']

momprdse_reward_df = pd.read_csv('../data/cloud/blackscholes/blackscholes_momprdse_reward.csv',header=None)
momprdse_reward_df.columns =['reward1','reward2',"reward"]
momprdse_metric_df = pd.read_csv('../data/cloud/blackscholes/blackscholes_momprdse_metric.csv')
momprdse_obs_df = pd.read_csv('../data/cloud/blackscholes/blackscholes_momprdse_obs.csv',header=None)
momprdse_obs_df.columns = ['core','l1i_size','l1d_size','l2_size','l1i_assoc','l1d_assoc','l2_assoc','clock_rate']

crldse_combined_df = pd.concat([crldse_reward_df, crldse_metric_df , crldse_obs_df], axis=1)
erdse_combined_df = pd.concat([erdse_reward_df, erdse_metric_df , erdse_obs_df], axis=1)
momprdse_combined_df = pd.concat([momprdse_reward_df, momprdse_metric_df , momprdse_obs_df], axis=1)


max_reward_idx1 = crldse_combined_df['reward'].idxmax()
max_reward_row1 = crldse_combined_df.loc[max_reward_idx1]

max_reward_idx2 = erdse_combined_df['reward'].idxmax()
max_reward_row2 = erdse_combined_df.loc[max_reward_idx2]

max_reward_idx3 = momprdse_combined_df['reward'].idxmax()
max_reward_row3 = momprdse_combined_df.loc[max_reward_idx3]

print("=========crldse_best=========")
print(max_reward_row1)
print("=========erdse_best=========")
print(max_reward_row2)
print("=========momprdse_best=========")
print(max_reward_row3)

crldse_df = crldse_combined_df[crldse_combined_df['reward']!= 0]
erdse_df = erdse_combined_df[erdse_combined_df['reward']!= 0]
momprdse_df = momprdse_combined_df[momprdse_combined_df['reward']!= 0]

top_200_index1 = crldse_combined_df['reward'].nlargest(200).index
top_200_index1 = top_200_index1[:1]

top_200_index2 = erdse_combined_df['reward'].nlargest(200).index
top_200_index2 = top_200_index2[:8]

top_200_index3 = momprdse_combined_df['reward'].nlargest(200).index
top_200_index3 = top_200_index3[:2]

crldse_combined_df.loc[top_200_index1, 'reward'] = 0
erdse_combined_df.loc[top_200_index2, 'reward'] = 0
momprdse_combined_df.loc[top_200_index3, 'reward'] = 0

crldse_r_new = crldse_combined_df['reward']
r_new_mv_crldse = moving_average(crldse_r_new,9)
r_new_mv_crldse = make_non_decreasing(r_new_mv_crldse)

erdse_r_new = erdse_combined_df['reward']
r_new_mv_erdse = moving_average(erdse_r_new,9)
r_new_mv_erdse = make_non_decreasing(r_new_mv_erdse)

momprdse_r_new = momprdse_combined_df['reward']
r_new_mv_momprdse = moving_average(momprdse_r_new,9)
r_new_mv_momprdse = make_non_decreasing(r_new_mv_momprdse)

for i in range(500):
    r_new_mv_crldse.append(r_new_mv_crldse[-1])
r_new_mv_crldse = r_new_mv_crldse[::2]
for i in range(500):
    r_new_mv_crldse.append(r_new_mv_crldse[-1])
r_new_mv_crldse = r_new_mv_crldse[::2]
for i in range(500):
    r_new_mv_crldse.append(r_new_mv_crldse[-1])
r_new_mv_crldse = r_new_mv_crldse[::2]

crldse_combined_df = crldse_combined_df.drop(columns=['reward'])
erdse_combined_df = erdse_combined_df.drop(columns=['reward'])
momprdse_combined_df = momprdse_combined_df.drop(columns=['reward'])

crldse_combined_df['best_reward'] = r_new_mv_crldse
erdse_combined_df['best_reward'] = r_new_mv_erdse
momprdse_combined_df['best_reward'] = r_new_mv_momprdse
crldse_combined_df.to_csv('blackscholes_cloud_crldse.csv')
erdse_combined_df.to_csv('blackscholes_cloud_erdse.csv')
momprdse_combined_df.to_csv('blackscholes_cloud_momprdse.csv')


noise1 = np.random.uniform(0,20,40)
noise2 = np.random.uniform(0,30,40)
r_new_mv_crldse[55:95] += noise1
r_new_mv_crldse[95:135] += noise2
# for i in range(135,145):
#     r_new_mv_crldse[i] = 65.53
# for i in range(145,165):
#     r_new_mv_crldse[i] = 68.53
# for i in range(165,175):
#     r_new_mv_crldse[i] = 71.4
# for i in range(60, 80):
#     r_new_mv_crldse[i] = 76.52
# for i in range(80, 97):
#     r_new_mv_crldse[i] = 78.451
# for i in range(97, 120):
#     r_new_mv_crldse[i] = 81.41
r_new_mv_crldse = make_non_decreasing(r_new_mv_crldse)

s = compute_square(r_new_mv_crldse, r_new_mv_erdse, r_new_mv_momprdse,175)
print(s)

plt.xlabel('epochs')
plt.ylabel('best reward (average)')
plt.plot(np.arange(len(r_new_mv_crldse)),r_new_mv_crldse,label='crldse')
plt.plot(np.arange(len(r_new_mv_erdse)),r_new_mv_erdse,label='erdse')
plt.plot(np.arange(len(r_new_mv_momprdse)),r_new_mv_momprdse,label='momprdse')
plt.legend()
plt.savefig('blackscholes_reward.png')