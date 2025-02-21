import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pareto_plot import pareto_front
from utils import make_non_decreasing, moving_average, compute_square

benchmark = 'canneal'
target = 'normal'

crldse_df = pd.read_csv(f'{benchmark}_{target}_crldse.csv')
erdse_df = pd.read_csv(f'{benchmark}_{target}_erdse.csv')
momprdse_df = pd.read_csv(f'{benchmark}_{target}_momprdse.csv')
nsga2_df = pd.read_csv(f'{benchmark}_{target}_nsga2.csv')
mopso_df = pd.read_csv(f'{benchmark}_{target}_mopso.csv')
sac_df = pd.read_csv(f'{benchmark}_{target}_sac.csv')
ppo_df = pd.read_csv(f'{benchmark}_{target}_ppo.csv')


crldse_reward = crldse_df['reward']
erdse_reward = erdse_df['reward']
momprdse_reward = momprdse_df['reward1']
nsga2_reward = nsga2_df['reward']
mopso_reward = mopso_df['reward']
sac_reward = sac_df['reward']
ppo_reward = ppo_df['reward']

#s = compute_square(r_new_mv_crldse, r_new_mv_erdse, r_new_mv_momprdse,120)

plt.xlabel('epochs')
plt.ylabel('best reward (average)')
plt.xlim(-5,505)
plt.ylim(-5,90)
plt.plot([120,120],[-5,82],color='black',linestyle='--')
plt.text(120, -8, '120')
plt.plot(np.arange(len(crldse_reward)),crldse_reward,label='crldse')
plt.plot(np.arange(len(erdse_reward)),erdse_reward,label='erdse')
plt.plot(np.arange(len(momprdse_reward)),momprdse_reward,label='momprdse')
plt.plot(np.arange(len(nsga2_reward)),nsga2_reward,label='nsga2')
plt.plot(np.arange(len(mopso_reward)),mopso_reward,label='mopso')
plt.plot(np.arange(len(sac_reward)),sac_reward,label='sac')
plt.plot(np.arange(len(ppo_reward)),ppo_reward,label='ppo')
plt.legend()
plt.savefig('reward.png')