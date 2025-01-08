import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pareto_plot import pareto_front
from utils import make_non_decreasing, moving_average, compute_square


crldse_df = pd.read_csv('../data/canneal_normal_crldse.csv')
erdse_df = pd.read_csv('../data/canneal_normal_erdse.csv')
momprdse_df = pd.read_csv('../data/canneal_normal_momprdse.csv')


r_new_mv_crldse = crldse_df['best_reward']
r_new_mv_erdse = erdse_df['best_reward']
r_new_mv_momprdse = momprdse_df['best_reward']

s = compute_square(r_new_mv_crldse, r_new_mv_erdse, r_new_mv_momprdse,120)
print(s)

plt.xlabel('epochs')
plt.ylabel('best reward (average)')
plt.xlim(-5,505)
plt.ylim(-5,90)
plt.plot([120,120],[-5,82],color='black',linestyle='--')
plt.text(120, -8, '120')
plt.plot(np.arange(len(r_new_mv_crldse)),r_new_mv_crldse,label='crldse')
plt.plot(np.arange(len(r_new_mv_erdse)),r_new_mv_erdse,label='erdse')
plt.plot(np.arange(len(r_new_mv_momprdse)),r_new_mv_momprdse,label='momprdse')
plt.legend()
plt.savefig('reward.png')