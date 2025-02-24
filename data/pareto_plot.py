import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def filter_latency(df):
    """
    筛选 DataFrame 中 latency 列的值在 0.0（不包含）到 1.0（包含）之间的行
    """
    df['latency'] = pd.to_numeric(df['latency'], errors='coerce')
    return df[(np.abs(df['latency'] - 0.0) > 0.00001) & (df['latency'] < 1.0)]

def euclidean_distance(point1, point2):
    return np.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)


def select_and_calculate_distances(pareto_fronts):
    # 合并所有帕累托前沿
    all_points = pd.concat(pareto_fronts, ignore_index=True)
    
    # 选择两个最好的点
    best_points = all_points.nsmallest(2, ['power', 'latency'])
    
    # 从每个帕累托前沿中选择前两个点
    selected_points = [front.iloc[:2] for front in pareto_fronts]
    
    # 计算截取的点与最好的两个点之间的距离
    distances = []
    for i, selected in enumerate(selected_points):
        total_distance = 0
        for j, point in selected.iterrows():
            for k, best_point in best_points.iterrows():
                distance = euclidean_distance(point, best_point)
                total_distance += distance
        distances.append((i, total_distance))
    
    return distances, best_points

def pareto_frontier(df: pd.DataFrame) -> pd.DataFrame:
    """
    计算帕累托前沿。
    :param df: pandas DataFrame, 形状为 (n,2)，列名为 ['power', 'latency']
    :return: 仅包含帕累托前沿点的 DataFrame
    """
    assert 'power' in df.columns and 'latency' in df.columns, "DataFrame 必须包含 'power' 和 'latency' 列"
    
    # 按照 power 升序，若 power 相同则按 latency 降序排序
    df_sorted = df.sort_values(by=['power', 'latency'], ascending=[True, False]).reset_index(drop=True)
    
    pareto_points = []
    min_latency = float('inf')
    
    # 遍历排序后的数据，筛选出帕累托前沿点
    for _, row in df_sorted.iterrows():
        if row['latency'] < min_latency:
            pareto_points.append(row)
            min_latency = row['latency']
    
    return pd.DataFrame(pareto_points)

def plot_pareto_front(data1, data2, data3,names):
    pareto_front1 = pareto_frontier(data1)
    pareto_front2 = pareto_frontier(data2)
    pareto_front3 = pareto_frontier(data3)
    d, bes_p = select_and_calculate_distances([pareto_front1,pareto_front2,pareto_front3])
    d = pd.DataFrame(d)
    print("ADRS:", d)

    name1, name2, name3 = names[0], names[1], names[2]
    
    x_label, y_label = data1.columns[:2]
    x_label, y_label = y_label, x_label
    
    plt.ylim(-5,60)
    combined_data = pd.concat([data1, data2, data3], ignore_index=True)
    plt.scatter(combined_data[x_label], combined_data[y_label], color='lightgray', label='all data',alpha=0.3)

    plt.scatter(pareto_front1[x_label], pareto_front1[y_label], color='red', marker='x', label=name1)
    plt.scatter(pareto_front2[x_label], pareto_front2[y_label], color='blue', marker='^', label=name2,facecolors='none')
    plt.scatter(pareto_front3[x_label], pareto_front3[y_label], color='green', marker='+', label=name3)
    
    plt.legend()
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.savefig('pareto.png', format='png',dpi=600)

def plot_pareto_front_all_data(data_list, names_list):
    # 计算每个数据集的帕累托前沿
    pareto_fronts = []
    for data,name in zip(data_list,names_list):
        cur_pareto = pareto_frontier(data)
        #print(f"{name}'s current pareto shape: ",cur_pareto.shape)
        pareto_fronts.append(cur_pareto)
    
    # print(pareto_fronts)

    with open("pareto_fronts.csv", 'w', newline='') as file:
        for i, pareto_front in enumerate(pareto_fronts):
            if i > 0:
                # 为不同的数据集添加分隔行
                file.write('\n')
            pareto_front.to_csv(file, mode='a', index=False)
    
    # 计算距离
    d, bes_p = select_and_calculate_distances(pareto_fronts)
    d = pd.DataFrame(d)
    d['name'] = names_list
    print("ADRS:", d)
    
    # 获取 x 和 y 标签
    x_label, y_label = data_list[0].columns[:2]
    x_label, y_label = y_label, x_label
    
    # 合并所有数据并绘制所有数据的散点图
    combined_data = pd.concat(data_list, ignore_index=True)
    # combined_data.to_csv('combined_data.csv', index=False)

    x_min = combined_data[x_label].min()
    x_max = combined_data[x_label].max()
    y_min = combined_data[y_label].min()
    y_max = combined_data[y_label].max()

    # 可以根据需求稍微扩展范围，避免数据点紧贴坐标轴
    x_padding = (x_max - x_min) * 0.05
    y_padding = (y_max - y_min) * 0.05
    x_min -= x_padding
    x_max += x_padding
    y_min -= y_padding
    y_max += y_padding

    # 设置 x 轴和 y 轴范围
    local_x_min = 0.0001
    local_x_max = 0.0002
    local_y_min = 0
    local_y_max = 10

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    ax1.scatter(combined_data[x_label], combined_data[y_label], color='lightgray', label='all data', alpha=0.3)
    ax2.scatter(combined_data[x_label], combined_data[y_label], color='lightgray', label='all data', alpha=0.3)
    
    # 定义颜色和标记列表
    colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
    markers = ['x', '^', '+', 's', 'd', 'o', '*', 'v', '<', '>']
    
    # 绘制每个数据集的帕累托前沿
    for i, (pareto_front, name) in enumerate(zip(pareto_fronts, names_list)):
        color = colors[i % len(colors)]
        marker = markers[i % len(markers)]
        ax1.scatter(pareto_front[x_label], pareto_front[y_label], color=color, marker=marker, label=name)
        ax2.scatter(pareto_front[x_label], pareto_front[y_label], color=color, marker=marker, label=name)
    
    ax1.set_xlim(x_min, x_max)
    ax1.set_ylim(y_min, y_max)
    ax1.set_xlabel(x_label)
    ax1.set_ylabel(y_label)
    # ax1.set_title("(a)", loc='center', pad=-20)
    ax1.text(0.00035, -25, "(a)")
    ax1.legend(loc='upper right')

    ax2.set_xlim(local_x_min, local_x_max)
    ax2.set_ylim(local_y_min, local_y_max)
    ax2.set_xlabel(x_label)
    ax2.set_ylabel(y_label)
    # ax2.set_title("(b)", loc='center', pad=-20)
    ax2.text(0.00015, -1.4, "(b)")
    ax2.legend(loc='upper right')

    plt.subplots_adjust(wspace=0.3)
    # 设置图例、坐标轴标签并保存图像

    plt.savefig('pareto_all.png', format='png', dpi=600)

if __name__ == "__main__":
    # load data
    metric_list = ["power","latency"]

    benchmark = 'canneal'
    target = 'normal'

    crldse_df = pd.read_csv(f'{benchmark}_{target}_crldse.csv')
    erdse_df = pd.read_csv(f'{benchmark}_{target}_erdse.csv')
    momprdse_df = pd.read_csv(f'{benchmark}_{target}_momprdse.csv')
    acdse_df = pd.read_csv(f'{benchmark}_{target}_acdse.csv')
    nsga2_df = pd.read_csv(f'{benchmark}_{target}_nsga2.csv')
    mopso_df = pd.read_csv(f'{benchmark}_{target}_mopso.csv')
    sac_df = pd.read_csv(f'{benchmark}_{target}_sac.csv')
    ppo_df = pd.read_csv(f'{benchmark}_{target}_ppo.csv')

    dataframes = [
        ('crldse_df', crldse_df),
        ('erdse_df', erdse_df),
        ('momprdse_df', momprdse_df),
        ('acdse_df', acdse_df),
        ('nsga2_df', nsga2_df),
        ('mopso_df', mopso_df),
        ('sac_df', sac_df),
        ('ppo_df', ppo_df)
    ]
    print(crldse_df.head())

    dataframes = [crldse_df, erdse_df, momprdse_df, acdse_df, nsga2_df, mopso_df, sac_df, ppo_df]
    names = ['crldse', 'erdse', 'momprdse', 'acdse', 'nsga2', 'mopso', 'sac', 'ppo']

    # 筛选数据
    all_data = [df[metric_list] for df in dataframes]
    all_metric_name_list = names
    all_metric_data_list = []
    for data, n in zip(all_data,names):
        print(f"====== current data {n} ======")
        print(data.info())
        filtered_df = filter_latency(data)
        print(filtered_df[filtered_df['latency'] > 1.0])
        all_metric_data_list.append(filtered_df)

    # 打印每个筛选后 DataFrame 的形状
    for i, df in enumerate(all_metric_data_list):
        print(f"data {i} shape: ", df.shape)

    # 调用绘制帕累托前沿的函数
    plot_pareto_front_all_data(all_metric_data_list, all_metric_name_list)