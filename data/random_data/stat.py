import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def calculate_frequency(reward_list):
    reward_series = pd.Series(reward_list)
    bins = list(range(0, 81, 10))
    labels = [f'({bins[i]}, {bins[i + 1]}]' for i in range(len(bins) - 1)]
    reward_cut = pd.cut(reward_series, bins=bins, labels=labels, right=True)
    frequency = reward_cut.value_counts()
    return frequency

def load_data(normal_path, cloud_path, embed_path):
    embedded_df = pd.read_csv(embed_path)
    normal_df = pd.read_csv(normal_path)
    cloud_df = pd.read_csv(cloud_path)
    # print("cloud head:\n", cloud_df.head())
    # print("normal head:\n", normal_df.head())
    # print("embedded head:\n", embedded_df.head())
    

    embedded_reward_list = embedded_df['reward']*2
    normal_reward_list = normal_df['reward']*2
    cloud_reward_list = cloud_df['reward']*2

    embedded_reward_list = embedded_reward_list.to_list()
    normal_reward_list = normal_reward_list.to_list()
    cloud_reward_list = cloud_reward_list.to_list()
    print(cloud_reward_list[:20])

    normal_frequency = calculate_frequency(normal_reward_list[0:800])
    cloud_frequency = calculate_frequency(cloud_reward_list[0:800])
    embed_frequency = calculate_frequency(embedded_reward_list[0:800])
    print("normal:",normal_frequency)
    print("cloud:",cloud_frequency)
    print("embed:",embed_frequency)

    return normal_frequency,cloud_frequency,embed_frequency

def plot_frequency(normal_frequency, cloud_frequency, embed_frequency):
#    categories = normal_frequency.index.tolist()  # 使用频率统计的索引作为类别标签

    # 将频率统计结果转换为列表
    # normal_counts = normal_frequency.values.tolist()
    # cloud_counts = cloud_frequency.values.tolist()
    # embed_counts = embed_frequency.values.tolist()
    categories = sorted(normal_frequency.index, key=lambda x: list(map(int, x.strip('()[]').split(','))))[::-1]
    normal_counts = [normal_frequency.get(cat, 0) for cat in categories]
    cloud_counts = [cloud_frequency.get(cat, 0) for cat in categories]
    embed_counts = [embed_frequency.get(cat, 0) for cat in categories]
    print("sum normal:",sum(normal_counts))
    print("sum cloud: ",sum(cloud_counts))
    print("sum embed: ",sum(embed_counts))

    # 设置柱状图的宽度
    bar_width = 0.25

    # 设置每个柱子的位置
    index = np.arange(len(categories))

    # 绘制柱状图
    plt.figure(figsize=(10, 6))
    plt.bar(index, cloud_counts, bar_width, edgecolor='black',label='cloud', color='#68BeD9')
    plt.bar(index + bar_width, normal_counts, bar_width, edgecolor='black',label='pc', color='#ED8D5A')
    plt.bar(index + 2 * bar_width, embed_counts, bar_width, edgecolor='black',label='embeded', color='#257D8B')

    # 添加标题和标签
    plt.xlabel('Frequency')
    plt.ylabel('Counts')
    # plt.title('Frequency Statistics of Normal, Cloud, and Embed Files')
    plt.xticks(index + bar_width, categories)
    plt.legend()
    def add_labels(ax, bars, color='black'):
        for bar in bars:
            height = bar.get_height()
            ax.annotate('{}'.format(height),
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom', color=color)

    add_labels(plt.gca(), plt.gca().patches)

    # 显示图形
    plt.savefig('reward_stat.png',dpi=600)

if __name__ == "__main__":
    embed_path = '/app/experiment_code/data/random_data/blackscholes_embed_random.csv'
    normal_path = '/app/experiment_code/data/random_data/blackscholes_normal_random.csv'
    cloud_path = '/app/experiment_code/data/random_data/blackscholes_cloud_random.csv'
    normal_frequency, cloud_frequency, embed_frequency = load_data(normal_path, cloud_path, embed_path)

    plot_frequency(normal_frequency, cloud_frequency, embed_frequency)



