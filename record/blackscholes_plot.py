import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pareto_plot import pareto_front
from utils import make_non_decreasing, moving_average, compute_square


crldse_df = pd.read_csv('../data/blackscholes_cloud_crldse.csv')
erdse_df = pd.read_csv('../data/blackscholes_cloud_erdse.csv')
momprdse_df = pd.read_csv('../data/blackscholes_cloud_momprdse.csv')


r_new_mv_crldse = crldse_df['best_reward']
r_new_mv_erdse = erdse_df['best_reward']
r_new_mv_momprdse = momprdse_df['best_reward']

s = compute_square(r_new_mv_crldse, r_new_mv_erdse, r_new_mv_momprdse,105)
print("reward square:",s)

plt.xlabel('epochs')
plt.ylabel('best reward (average)')
plt.xlim(-5,505)
plt.ylim(-5,90)
plt.plot([105,105],[-5,62],color='black',linestyle='--')
plt.text(105, -8, '105')
plt.plot(np.arange(len(r_new_mv_crldse)),r_new_mv_crldse,label='crldse')
plt.plot(np.arange(len(r_new_mv_erdse)),r_new_mv_erdse,label='erdse')
plt.plot(np.arange(len(r_new_mv_momprdse)),r_new_mv_momprdse,label='momprdse')
plt.legend()
plt.savefig('blackscholes_reward.png')