import os
import pandas as pd
import matplotlib.pyplot as plt

folder_path = '/app/experiment_code/data'

all_reward_list = []

# 初始化两个空列表来分别存储 normal 和 cloud 文件的 reward 值
normal_reward_list = []
cloud_reward_list = []

# 遍历文件夹中的所有文件
for filename in os.listdir(folder_path):
    if filename.endswith('.csv'):
        # 构建文件的完整路径
        file_path = os.path.join(folder_path, filename)
        try:
            # 读取 CSV 文件
            df = pd.read_csv(file_path)
            # 检查是否存在 'reward' 列
            if 'reward' in df.columns:
                # 提取 'reward' 列的值，过滤掉负数和 0
                valid_rewards = df[df['reward'] > 0]['reward'].tolist()
                if 'normal' in filename:
                    normal_reward_list.extend(valid_rewards)
                elif 'cloud' in filename:
                    cloud_reward_list.extend(valid_rewards)
        except Exception as e:
            print(f"读取文件 {filename} 时出错: {e}")

# 定义 0 - 100 分 10 个数据段
bins = list(range(0, 101, 10))
labels = [f'({bins[i]}, {bins[i + 1]}]' for i in range(len(bins) - 1)]

# 定义一个函数来进行分段和频率统计
def calculate_frequency(reward_list):
    reward_series = pd.Series(reward_list)
    reward_cut = pd.cut(reward_series, bins=bins, labels=labels, right=True)
    frequency = reward_cut.value_counts()
    return frequency

# 分别计算 normal 和 cloud 文件的频率
normal_frequency = calculate_frequency(normal_reward_list)
cloud_frequency = calculate_frequency(cloud_reward_list)

print("Normal 文件的频率统计：")
print(normal_frequency)
print("\nCloud 文件的频率统计：")
print(cloud_frequency)



# 定义 Normal 文件的频率统计数据
normal_data = {
    '(0, 10]': 2051,
    '(70, 80]': 982,
    '(20, 30]': 957,
    '(30, 40]': 617,
    '(50, 60]': 506,
    '(10, 20]': 481,
    '(60, 70]': 293,
    '(40, 50]': 223,
    '(80, 90]': 0,
    '(90, 100]': 0
}
# 删除值为 0 的区间
normal_data = {k: v for k, v in normal_data.items() if k not in ['(80, 90]', '(90, 100]']}
normal_series = pd.Series(normal_data)

# 定义 Cloud 文件的频率统计数据
cloud_data = {
    '(30, 40]': 1511,
    '(20, 30]': 1392,
    '(70, 80]': 993,
    '(0, 10]': 881,
    '(50, 60]': 679,
    '(10, 20]': 476,
    '(60, 70]': 330,
    '(40, 50]': 276,
    '(80, 90]': 0,
    '(90, 100]': 0
}
# 删除值为 0 的区间
cloud_data = {k: v for k, v in cloud_data.items() if k not in ['(80, 90]', '(90, 100]']}
cloud_series = pd.Series(cloud_data)

# 确保两个 Series 的索引顺序一致
index_order = ['(0, 10]', '(10, 20]', '(20, 30]', '(30, 40]', '(40, 50]',
               '(50, 60]', '(60, 70]', '(70, 80]']
# 反转索引顺序
reversed_index_order = index_order[::-1]
normal_series = normal_series.reindex(reversed_index_order)
cloud_series = cloud_series.reindex(reversed_index_order)

# 设置图片清晰度
plt.rcParams['figure.dpi'] = 300

# 设置图形大小
plt.figure(figsize=(12, 6))

# 绘制柱状图
bar_width = 0.35
index = range(len(reversed_index_order))
plt.bar(index, normal_series, bar_width, label='Normal 文件')
plt.bar([i + bar_width for i in index], cloud_series, bar_width, label='Cloud 文件')

# 设置坐标轴标签和标题
plt.xlabel('数据段')
plt.ylabel('频率')
plt.title('Normal 文件和 Cloud 文件的频率统计柱状图')
plt.xticks([i + bar_width / 2 for i in index], reversed_index_order, rotation=45)

# 添加图例
plt.legend()

# 显示图形
plt.tight_layout()
plt.savefig('stat.png')