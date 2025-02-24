import pandas as pd
import numpy as np
from sklearn import manifold
import matplotlib.pyplot as plt
import os

# 定义 tsne3D 函数
def tsne3D(vector_list, reward_list, method):
    action_array = np.array(vector_list)
    reward_continue_array = np.array(reward_list)

    tsne = manifold.TSNE(n_components=2, init="pca", random_state=501)
    print(f"Start to load t-SNE")
    x_tsne = tsne.fit_transform(action_array)

    x_min, x_max = x_tsne.min(0), x_tsne.max(0)
    x_norm = (x_tsne - x_min) / (x_max - x_min)

    fig_3D = plt.figure()
    tSNE_3D = plt.axes(projection='3d')
    tSNE_3D.scatter3D(x_norm[:, 0], x_norm[:, 1], reward_continue_array, c=reward_continue_array, vmax=20, cmap="rainbow", alpha=0.5)
    tSNE_3D.set_xlabel("x")
    tSNE_3D.set_ylabel("y")
    tSNE_3D.set_zlabel("Reward")
    tSNE_3D.set_zlim((0, 80))
    tSNE_3D.set_zticks([0, 20, 40, 60, 80])
    fname = method + "_" + "tSEN_3D" + ".png"
    fig_3D.savefig(fname, format="png")


path = '/app/experiment_code/data'

# 初始化一个空列表，用于存储读取的 DataFrame
data_frames = []

# 遍历指定路径下的所有文件
for filename in os.listdir(path):
    if filename.endswith('.csv') and 'cloud' in filename and 'acdse' not in filename and 'momprdse' not in filename:
        file_path = os.path.join(path, filename)
        try:
            # 读取 CSV 文件
            df = pd.read_csv(file_path)
            data_frames.append(df)
        except Exception as e:
            print(f"read file {filename} error: {e}")

# 将所有 DataFrame 堆叠起来
if data_frames:
    stacked_data = pd.concat(data_frames, ignore_index=True)
    stacked_data = stacked_data[stacked_data['latency'] < 999.0]
    stacked_data = stacked_data.dropna()
    stacked_data.info()
    stacked_data.head()
    # 分离 reward 列和其他列的数据
    reward_list = stacked_data['reward'].tolist()
    vector_list = stacked_data[['latency','Area','energy','power']].values.tolist()
    # 调用 tsne3D 函数
    method = 'cloud'
    tsne3D(vector_list, reward_list, method)

