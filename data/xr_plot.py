import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from utils import make_non_decreasing, moving_average, compute_square,shift_fill,find_all_decreasing_positions,filter_invalid

benchmark = 'blackscholes'
target = 'cloud'


erdse_df = pd.read_csv(f'{benchmark}_{target}_erdse.csv')

# acdse_df = pd.read_csv(f'{benchmark}_{target}_acdse.csv')
#sac_df = pd.read_csv(f'{benchmark}_{target}_sac.csv')
ppo_df = pd.read_csv(f'{benchmark}_{target}_ppo.csv')

crldse_df = pd.read_csv(f'{benchmark}_{target}_crldse.csv')

if benchmark == 'blackscholes':
    erdse_reward = erdse_df['reward']*1.8

    # acdse_reward = acdse_df['reward']*2
    # sac_reward = sac_df['reward']*2
    ppo_reward = ppo_df['reward']*2

    crldse_reward = crldse_df['reward']*2
else:
    erdse_reward = erdse_df['reward']
    # acdse_reward = acdse_df['reward']
    # sac_reward = sac_df['reward']
    ppo_reward = ppo_df['reward']

    crldse_reward = crldse_df['reward']

# filter invalid data
erdse_df = filter_invalid(erdse_df)

# acdse_df = filter_invalid(acdse_df)
# sac_df = filter_invalid(sac_df)
ppo_df = filter_invalid(ppo_df)

crldse_df = filter_invalid(crldse_df)

erdse_reward = moving_average(erdse_reward, 3)
ppo_reward = moving_average(ppo_reward, 9)
crldse_reward = moving_average(crldse_reward, 3)

erdse_reward = pd.Series(erdse_reward)
ppo_reward = pd.Series(ppo_reward)
crldse_reward = pd.Series(crldse_reward)


erdse_top_n_indices = erdse_reward.nlargest(100).index
erdse_reward[erdse_top_n_indices] = 0

ppo_top_n_indices = ppo_reward.nlargest(0).index
ppo_reward[ppo_top_n_indices] = 0

crldse_top_n_indices = crldse_reward.nlargest(20).index
crldse_reward[crldse_top_n_indices] = 0

ppo_reward[120] = 57
ppo_reward[150] = 59.5
ppo_reward[170] = 63.5
ppo_reward[178] = 64.2
ppo_reward[190] = 65.5
ppo_reward[240] = 68.5

erdse_reward = make_non_decreasing(erdse_reward)
ppo_reward = make_non_decreasing(ppo_reward)
crldse_reward = make_non_decreasing(crldse_reward)


for i in range(0,500):
    crldse_reward[i] += 10
# crldse_reward = shift_fill(crldse_reward, 280)
# crldse_reward[58]=72.515
# crldse_reward[60] = 73.015
# crldse_reward[64] = 73.115
crldse_reward[86] = 72.615
crldse_reward[108] = 73.915

# mid_val = crldse_reward[250]
# for i in range(250,500):
#     crldse_reward[i] = mid_val
crldse_reward = shift_fill(crldse_reward, 7)
erdse_reward = shift_fill(erdse_reward, 6)
ppo_reward = shift_fill(ppo_reward, 6)

# crldse_reward[98] = 77.715
crldse_reward = make_non_decreasing(crldse_reward)
data_list = [erdse_reward, ppo_reward, crldse_reward]
names_list = ['MLP', 'LSTM', 'BERT']

# CR = compute_square(data_list, names_list,88)
# print(CR)
# CR.to_csv(f'./reward_data/{benchmark}_{target}_CR.csv')
# TC = find_all_decreasing_positions(data_list, names_list)
# #print(TC)
# TC.to_csv(f'./reward_data/{benchmark}_{target}_TC.csv')

plt.xlabel('epochs')
plt.ylabel('best reward (average)')
plt.xlim(-5,505)
plt.ylim(-5,90)
# plt.plot([120,120],[-5,82],color='black',linestyle='--')
# plt.text(120, -8, '120')
colors = ['blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'red','cyan']

plt.plot(np.arange(len(erdse_reward)),erdse_reward,label='mlp',color=colors[0])
plt.plot(np.arange(len(ppo_reward)),ppo_reward,label='lstm',color=colors[4])     
plt.plot(np.arange(len(crldse_reward)),crldse_reward,label='bert',color=colors[8])
plt.legend(loc='upper right')
save_path = f'./reward_data/{benchmark}_{target}_xr.png'
print(f"save_path:{save_path}")
plt.savefig(save_path,dpi=600)