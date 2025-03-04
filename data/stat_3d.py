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

    scatter = ax.scatter3D(x_norm[:, 0], x_norm[:, 1], c=reward_continue_array, vmax=60, cmap="rainbow", alpha=0.3)
    ax.scatter3D(x_norm[:, 0], x_norm[:, 1], reward_continue_array, c=reward_continue_array, vmax=80, cmap="rainbow", alpha=0.3)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("Reward")
    ax.set_zlim((0, 80))
    ax.set_zticks([0, 20, 40, 60, 80])

    # 去掉坐标轴边框
    # for edge, spine in ax.spines.items():
    #     spine.set_visible(False)

    return scatter


embed_path = '/app/experiment_code/data/blackscholes_cloud_crldse.csv'
normal_path = '/app/experiment_code/data/blackscholes_cloud_ppo.csv'
cloud_path = '/app/experiment_code/data/blackscholes_cloud_momprdse_new.csv'

# 初始化一个空列表，用于存储读取的 DataFrame
embed_df = pd.read_csv(embed_path)
normal_df = pd.read_csv(normal_path)
cloud_df = pd.read_csv(cloud_path)



# Filter and prepare data for each method
embed_filted_data = embed_df[embed_df['latency'] < 9990.].dropna()
normal_filted_data = normal_df[normal_df['latency'] < 999.0].dropna()
cloud_filted_data = cloud_df[cloud_df['latency'] < 999.0].dropna()

# Prepare the data for t-SNE
vector_list_name = ["core", "l1i_size", "l1d_size", "l2_size", "l1i_assoc", "l1d_assoc", "l2_assoc", "clock_rate"]
embed_reward_list = embed_filted_data['reward']*2
embed_reward_list = embed_reward_list.tolist()
embed_vector_list = embed_filted_data[vector_list_name].values.tolist()

normal_reward_list = normal_filted_data['reward']*2
normal_reward_list = normal_reward_list.tolist()
normal_vector_list = normal_filted_data[vector_list_name].values.tolist()

cloud_reward_list = cloud_filted_data['reward']*2
cloud_reward_list = cloud_reward_list.tolist()
cloud_vector_list = cloud_filted_data[vector_list_name].values.tolist()

# Create a figure with three subplots
fig = plt.figure(figsize=(18, 6))

ax1 = fig.add_subplot(131, projection='3d')
ax1.set_title('(a) crldse')
scatter1 = tsne3D(embed_vector_list, embed_reward_list, ax1, 'Embed')

ax2 = fig.add_subplot(132, projection='3d')
ax2.set_title('(b) ppo')
scatter2 = tsne3D(normal_vector_list, normal_reward_list, ax2, 'Normal')

ax3 = fig.add_subplot(133, projection='3d')
ax3.set_title('(c) momprdse')
scatter3 = tsne3D(cloud_vector_list, cloud_reward_list, ax3, 'Cloud')

# Save the figure
fpath = './random_data/tsne_plots.png'
plt.savefig(fpath, format="png",dpi=600)

