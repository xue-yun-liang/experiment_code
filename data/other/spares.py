import pandas as pd
import matplotlib.pyplot as plt
from utils import make_non_decreasing

# 读取数据
data1 = pd.read_csv('/app/experiment_code/data/blackscholes_cloud_acdse.csv',index_col=0)
data2 = pd.read_csv('/app/experiment_code/data/blackscholes_cloud_crldse_new.csv',index_col=0)

data1['best reward'] = make_non_decreasing(data1['reward'])
data2['best reward'] = make_non_decreasing(data2['reward'])

# 创建图形和子图
fig, axes = plt.subplots(1, 2, figsize=(12, 6))

# 绘制第一个子图
axes[0].plot(data1['reward'], alpha=0.3, label='real reward')
axes[0].plot(data1['best reward'], color='red', label='best reward')
axes[0].set_title('without HER')
axes[0].legend()

# 绘制第二个子图
axes[1].plot(data2['reward'], alpha=0.3, label='real reward')
axes[1].plot(data2['best reward'], color='red', label='best reward')
axes[1].set_title('with HER')
axes[1].legend()

# 调整布局
plt.tight_layout()
plt.savefig('spares.png', dpi=600)