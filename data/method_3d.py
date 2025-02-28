import pandas as pd
import numpy as np
from sklearn import manifold
import matplotlib.pyplot as plt
import os

# 定义 tsne3D 函数
def tsne3D(vector_list, reward_list, ax, method):
    action_array = np.array(vector_list)
    reward_continue_array = np.array(reward_list)

    tsne = manifold.TSNE(n_components=2, init="pca", random_state=501)
    print(f"Start to load t-SNE to plot for {method}")
    x_tsne = tsne.fit_transform(action_array)

    x_min, x_max = x_tsne.min(0), x_tsne.max(0)
    x_norm = (x_tsne - x_min) / (x_max - x_min)

    scatter = ax.scatter3D(x_norm[:, 0], x_norm[:, 1], c=reward_continue_array, vmax=20, cmap="rainbow", alpha=0.3)
    ax.scatter3D(x_norm[:, 0], x_norm[:, 1], reward_continue_array, c=reward_continue_array, vmax=20, cmap="rainbow", alpha=0.3)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("Reward")
    ax.set_zlim((0, 80))
    ax.set_zticks([0, 20, 40, 60, 80])

    # 去掉坐标轴边框
    # for edge, spine in ax.spines.items():
    #     spine.set_visible(False)

    return scatter

target = 'cloud'
benchmark = 'blackscholes'

path1 = f'/app/experiment_code/data/{benchmark}_{target}_crldse.csv'
path2 = f'/app/experiment_code/data/{benchmark}_{target}_ppo.csv'
path3 = f'/app/experiment_code/data/{benchmark}_{target}_momprdse_new.csv'

# 初始化一个空列表，用于存储读取的 DataFrame
df1 = pd.read_csv(path1)
df2 = pd.read_csv(path2)
df3 = pd.read_csv(path3)


# Filter and prepare data for each method
filted_data1 = df1[df1['latency'] < 9990.].dropna()
filted_data2 = df2[df2['latency'] < 999.0].dropna()
filted_data3 = df3[df3['latency'] < 999.0].dropna()

# Prepare the data for t-SNE
vector_list_name = ["core", "l1i_size", "l1d_size", "l2_size", "l1i_assoc", "l1d_assoc", "l2_assoc", "clock_rate"]
reward_list1 = filted_data1['reward']*2
reward_list2 = reward_list1.tolist()
vector_list1 = filted_data1[vector_list_name].values.tolist()

reward_list2 = filted_data2['reward']*2
reward_list2 = reward_list2.tolist()
vector_list2 = filted_data2[vector_list_name].values.tolist()

reward_list3 = filted_data3['reward']*2
reward_list3 = reward_list3.tolist()
vector_list3 = filted_data3[vector_list_name].values.tolist()

# Create a figure with three subplots
fig = plt.figure(figsize=(18, 6))

ax1 = fig.add_subplot(131, projection='3d')
ax1.set_title('(a) crldse')
scatter1 = tsne3D(vector_list1, reward_list1, ax1, 'Embed')

ax2 = fig.add_subplot(132, projection='3d')
ax2.set_title('(b) ppo')
scatter2 = tsne3D(vector_list2, reward_list2, ax2, 'Normal')

ax3 = fig.add_subplot(133, projection='3d')
ax3.set_title('(c) momprdse')
scatter3 = tsne3D(vector_list3, reward_list3, ax3, 'Cloud')

# Save the figure
fpath = './method_tsne_plots.png'
plt.savefig(fpath, format="png",dpi=600)

