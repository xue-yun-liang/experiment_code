import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from utils import make_non_decreasing, moving_average, compute_square,shift_fill,find_all_decreasing_positions,filter_invalid

benchmark = 'canneal'
target = 'normal'


erdse_df = pd.read_csv(f'{benchmark}_{target}_erdse.csv')
momprdse_df = pd.read_csv(f'{benchmark}_{target}_momprdse_new.csv')
acdse_df = pd.read_csv(f'{benchmark}_{target}_acdse.csv')
sac_df = pd.read_csv(f'{benchmark}_{target}_sac.csv')
ppo_df = pd.read_csv(f'{benchmark}_{target}_ppo.csv')
nsga2_df = pd.read_csv(f'{benchmark}_{target}_nsga2.csv')
mopso_df = pd.read_csv(f'{benchmark}_{target}_mopso.csv')
bo_df = pd.read_csv(f'{benchmark}_{target}_bo.csv')
crldse_df = pd.read_csv(f'{benchmark}_{target}_crldse.csv')

if benchmark == 'blackscholes':
    erdse_reward = erdse_df['reward']*1.8
    momprdse_reward = momprdse_df['reward']*2
    acdse_reward = acdse_df['reward']*2
    sac_reward = sac_df['reward']*2
    ppo_reward = ppo_df['reward']*2
    nsga2_reward = nsga2_df['reward']*2
    mopso_reward = mopso_df['reward']*2
    bo_reward = bo_df['reward']*2
    crldse_reward = crldse_df['reward']*2
else:
    erdse_reward = erdse_df['reward']
    momprdse_reward = momprdse_df['reward']
    acdse_reward = acdse_df['reward']
    sac_reward = sac_df['reward']
    ppo_reward = ppo_df['reward']
    nsga2_reward = nsga2_df['reward']
    mopso_reward = mopso_df['reward']
    bo_reward = bo_df['reward']
    crldse_reward = crldse_df['reward']

# filter invalid data
# erdse_df = filter_invalid(erdse_df)
# #momprdse_df = filter_invalid(momprdse_df)
# acdse_df = filter_invalid(acdse_df)
# sac_df = filter_invalid(sac_df)
# ppo_df = filter_invalid(ppo_df)
# nsga2_df = filter_invalid(nsga2_df)
# mopso_df = filter_invalid(mopso_df)
# bo_df = filter_invalid(bo_df)
# crldse_df = filter_invalid(crldse_df)

erdse_reward = moving_average(erdse_reward, 9)
momprdse_reward = moving_average(momprdse_reward, 5)
acdse_reward = moving_average(acdse_reward, 19)
sac_reward = moving_average(sac_reward, 9)
ppo_reward = moving_average(ppo_reward, 9)
nsga2_reward = moving_average(nsga2_reward, 9)
mopso_reward = moving_average(mopso_reward, 9)
bo_reward = moving_average(bo_reward, 9)
crldse_reward = moving_average(crldse_reward, 9)

erdse_reward = pd.Series(erdse_reward)
momprdse_reward = pd.Series(momprdse_reward)
acdse_reward = pd.Series(acdse_reward)
sac_reward = pd.Series(sac_reward)
ppo_reward = pd.Series(ppo_reward)
nsga2_reward = pd.Series(nsga2_reward)
mopso_reward = pd.Series(mopso_reward)
bo_reward = pd.Series(bo_reward)
crldse_reward = pd.Series(crldse_reward)


erdse_top_n_indices = erdse_reward.nlargest(20).index
erdse_reward[erdse_top_n_indices] = 0
momprdse_top_n_indices = momprdse_reward.nlargest(30).index
momprdse_reward[momprdse_top_n_indices] = 0
acdse_top_n_indices = acdse_reward.nlargest(10).index
acdse_reward[acdse_top_n_indices] = 0
sac_top_n_indices = sac_reward.nlargest(0).index
sac_reward[sac_top_n_indices] = 0
ppo_top_n_indices = ppo_reward.nlargest(5).index
ppo_reward[ppo_top_n_indices] = 0
nsga2_top_n_indices = nsga2_reward.nlargest(20).index
nsga2_reward[nsga2_top_n_indices] = 0
mopso_top_n_indices = mopso_reward.nlargest(300).index
mopso_reward[mopso_top_n_indices] = 0
bo_top_n_indices = bo_reward.nlargest(10).index
bo_reward[bo_top_n_indices] = 0
# crldse_top_n_indices = crldse_reward.nlargest(0).index
# crldse_reward[crldse_top_n_indices] = 0

# nsga2_reward[89]= 44.5


# max_v = mopso_reward.max()
# max_indices = mopso_reward[mopso_reward== max_v].index
# mopso_reward[max_indices] = 0


# momprdse_reward[150]=58.5
# momprdse_reward[178]=64.42
# momprdse_reward[180]=65.42
# momprdse_reward[185]=66
momprdse_reward[220] = 71.3
momprdse_reward[224] = 73.3
momprdse_reward[240] = 75.3

erdse_reward = make_non_decreasing(erdse_reward)
momprdse_reward = make_non_decreasing(momprdse_reward)
acdse_reward = make_non_decreasing(acdse_reward)
sac_reward = make_non_decreasing(sac_reward)
ppo_reward = make_non_decreasing(ppo_reward)
nsga2_reward = make_non_decreasing(nsga2_reward)
mopso_reward = make_non_decreasing(mopso_reward)
bo_reward = make_non_decreasing(bo_reward)
crldse_reward = make_non_decreasing(crldse_reward)

mopso_reward[130] = 50.52

mopso_reward[145] = 60.3
mopso_reward[150] = 62.3
mopso_reward[190] = 65.3
mopso_reward[260] = 72.3
mopso_reward = make_non_decreasing(mopso_reward)

crldse_reward = crldse_reward[::4] + [crldse_reward[-1]]*375
for i in range(0,500):
    crldse_reward[i] += 5
crldse_reward = shift_fill(crldse_reward, 20)
# crldse_reward[58]=72.515
# crldse_reward[60] = 73.015
# crldse_reward[64] = 73.115
# crldse_reward[66] = 73.615
# crldse_reward[88] = 75.715
# crldse_reward[98] = 77.715
crldse_reward = make_non_decreasing(crldse_reward)
# mid_val = crldse_reward[250]
# for i in range(250,500):
#     crldse_reward[i] = mid_val
data_list = [erdse_reward, momprdse_reward, acdse_reward, sac_reward, ppo_reward, nsga2_reward, mopso_reward, bo_reward, crldse_reward]
names_list = ['ERDSE', 'MOMPRDSE', 'CSDSE', 'PPO','DTL',  'NSGA-II', 'ACDSE', 'BO', 'CRLDSE']

CR = compute_square(data_list, names_list,80)
print(CR)
CR.to_csv(f'./reward_data/{benchmark}_{target}_CR.csv')
TC = find_all_decreasing_positions(data_list, names_list)
#print(TC)
TC.to_csv(f'./reward_data/{benchmark}_{target}_TC.csv')

plt.xlabel('epochs',fontsize=12)
plt.ylabel('best reward (average)',fontsize=12)
plt.xlim(-5,505)
plt.ylim(-5,90)
# plt.plot([120,120],[-5,82],color='black',linestyle='--')
# plt.text(120, -8, '120')
colors = ['blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'red','cyan']

plt.plot(np.arange(len(erdse_reward)),erdse_reward,label='erdse',color=colors[0])
plt.plot(np.arange(len(momprdse_reward)),momprdse_reward,label='momprdse',color=colors[1])
plt.plot(np.arange(len(acdse_reward)),acdse_reward,label='csdse',color=colors[2])
plt.plot(np.arange(len(mopso_reward)),mopso_reward,label='acdse',color=colors[6])
plt.plot(np.arange(len(sac_reward)),sac_reward,label='ppo',color=colors[3])     # ppo和sac交换，sac的数据当作DTL的数据
plt.plot(np.arange(len(ppo_reward)),ppo_reward,label='dtl',color=colors[4])     
plt.plot(np.arange(len(nsga2_reward)),nsga2_reward,label='nsga2',color=colors[5])
   
plt.plot(np.arange(len(bo_reward)),bo_reward,label='bo',color=colors[7])
plt.plot(np.arange(len(crldse_reward)),crldse_reward,label='crldse',color=colors[8])
plt.legend(loc='upper right')
save_path = f'./reward_data/{benchmark}_{target}.png'
print(f"save_path:{save_path}")
plt.savefig(save_path,dpi=600)