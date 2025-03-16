import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

metric_list = ["power","latency"]
metric_str = "_".join(metric_list)

benchmark = 'canneal'
target = 'normal'

def filter_invalid(df):
    df['power'] = pd.to_numeric(df['power'], errors='coerce')
    return df[(np.abs(df['power'] - 0.0) > 0.00001) & (df['power'] < 999.0)]

def euclidean_distance(point1, point2):
    return np.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)


def select_and_calculate_distances(pareto_fronts):
    # 合并所有帕累托前沿
    all_points = pd.concat(pareto_fronts, ignore_index=True)
    
    # 选择最好的点
    min_data_count = min(len(front) for front in pareto_fronts)
    print(f"***current best data count:{min_data_count}***")
    best_points = all_points.nsmallest(min_data_count, metric_list)
    
    # 从每个帕累托前沿中选择前两个点
    selected_points = [front.iloc[:min_data_count] for front in pareto_fronts]
    
    # 计算截取的点与最好的两个点之间的距离
    distances = []
    for i, selected in enumerate(selected_points):
        total_distance = 0
        for j, point in selected.iterrows():
            for k, best_point in best_points.iterrows():
                distance = euclidean_distance(point, best_point)
                total_distance += distance
        distances.append(total_distance)
    
    return distances, best_points

def pareto_frontier(df: pd.DataFrame) -> pd.DataFrame:
    """
    计算帕累托前沿。
    :param df: pandas DataFrame, 形状为 (n,2)，列名为 metric_list
    :return: 仅包含帕累托前沿点的 DataFrame
    """
 
    # 按照 power 升序，若 power 相同则按 latency 降序排序
    df_sorted = df.sort_values(by=metric_list, ascending=[True, False]).reset_index(drop=True)
    
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
    print("ADRS:",d)
    d.to_csv(f'./pareto_data/ADRS_{benchmark}_{target}_{metric_str}.csv')

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
    pareto_fronts_path = f"./pareto_data/{benchmark}_{target}_pareto_fronts.csv"
    print(f"saving {benchmark}_{target}_pareto_fronts.csv")
    with open(pareto_fronts_path, 'w', newline='') as file:
        for i, pareto_front in enumerate(pareto_fronts):

                # 为不同的数据集添加分隔行
            file.write(f'{names_list[i]}\n')
            pareto_front.to_csv(file, mode='a', index=False)
    
    # 计算距离
    d, bes_p = select_and_calculate_distances(pareto_fronts)
    adrs = pd.DataFrame(names_list, columns=['name'])
    adrs['ADRS'] = d
    print("ADRS:", adrs)
    adrs.to_csv(f'./pareto_data/ADRS_{benchmark}_{target}_{metric_str}.csv')
    
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
    local_x_min = 0.1
    local_x_max = 0.2
    local_y_min = 0
    local_y_max = 5

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    ax1.scatter(combined_data[x_label], combined_data[y_label], color='lightgray', label='all data', alpha=0.3)
    ax2.scatter(combined_data[x_label], combined_data[y_label], color='lightgray', label='all data', alpha=0.3, s=100)
    
    # 定义颜色和标记列表
    colors = ['blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'red','cyan']
    markers = ['^', '+', 's', 'd', 'o', '*', 'v', '<', 'x', '>']
    
    # 绘制每个数据集的帕累托前沿
    for i, (pareto_front, name) in enumerate(zip(pareto_fronts, names_list)):
        color = colors[i % len(colors)]
        marker = markers[i % len(markers)]
        # if name =='crldse':
        #     ax1.scatter(pareto_front[x_label], pareto_front[y_label]-0.23, color=color, marker=marker, label=name)
        #     ax2.scatter(pareto_front[x_label], pareto_front[y_label]-0.23, color=color, marker=marker, label=name, s=100)
        # else:
        ax1.scatter(pareto_front[x_label], pareto_front[y_label], color=color, marker=marker, label=name)
        ax2.scatter(pareto_front[x_label], pareto_front[y_label], color=color, marker=marker, label=name, s=100)

    if x_label == "latency":
        x_label = "latency(ms)"
    ax1.set_xlim(x_min, x_max)
    ax1.set_ylim(y_min, y_max)
    ax1.set_xlabel(x_label,fontsize=16)
    ax1.set_ylabel(y_label,fontsize=16)
    ax1.set_title("(a)", loc='center', pad=-20)
    ax1.legend(loc='upper right')

    ax2.set_xlim(local_x_min, local_x_max)
    ax2.set_ylim(local_y_min, local_y_max)
    ax2.set_xlabel(x_label,fontsize=16)
    ax2.set_ylabel(y_label,fontsize=16)
    ax2.set_title("(b)", loc='center', pad=-20)
    ax2.legend(loc='upper right')

    plt.subplots_adjust(wspace=0.3)
    # 设置图例、坐标轴标签并保存图像
    fig_path = f'./pareto_data/pareto_{benchmark}_{target}_{metric_str}.png'
    print("saveing", fig_path)
    plt.savefig(fig_path, format='png', dpi=600)

if __name__ == "__main__":
    # load data


    
    erdse_df = pd.read_csv(f'{benchmark}_{target}_erdse.csv')
    momprdse_df = pd.read_csv(f'{benchmark}_{target}_momprdse.csv')
    acdse_df = pd.read_csv(f'{benchmark}_{target}_acdse.csv')
    sac_df = pd.read_csv(f'{benchmark}_{target}_sac.csv')
    ppo_df = pd.read_csv(f'{benchmark}_{target}_ppo.csv')
    nsga2_df = pd.read_csv(f'{benchmark}_{target}_nsga2.csv')
    mopso_df = pd.read_csv(f'{benchmark}_{target}_mopso.csv')
    bo_df = pd.read_csv(f'{benchmark}_{target}_bo.csv')
    crldse_df = pd.read_csv(f'{benchmark}_{target}_crldse.csv')

    # dataframes = [
    #     ('erdse_df', erdse_df),
    #     ('momprdse_df', momprdse_df),
    #     ('acdse_df', acdse_df),
    #     ('sac_df', sac_df),
    #     ('ppo_df', ppo_df),
    #     ('nsga2_df', nsga2_df),
    #     ('mopso_df', mopso_df),
    #     ('bo_df', bo_df),
    #     ('crldse_df', crldse_df),
    # ]
    # print(crldse_df.head())
    

    dataframes = [erdse_df, momprdse_df, acdse_df, mopso_df, sac_df, ppo_df, nsga2_df,  bo_df, crldse_df]
    names_ = ['erdse', 'momprdse', 'acdse', 'csdse','ppo', 'dtl', 'nsga2',  'bo', 'crldse']
    names = ['crldse', 'erdse', 'momprdse', 'acdse', 'nsga2', 'mopso', 'sac', 'ppo']


    # 筛选数据
    all_data = [df[metric_list] for df in dataframes]
    all_metric_name_list = names_
    all_metric_data_list = []
    if metric_list[1] == 'latency':
        for data in all_data:
            data['latency'] = data['latency'] * 1000

    for data, n in zip(all_data,names_):
        print(f"====== current data {n} ======")
        print(data.info())
        filtered_df = filter_invalid(data)
        all_metric_data_list.append(filtered_df)

    # 打印每个筛选后 DataFrame 的形状
    for i, df in enumerate(all_metric_data_list):
        print(f"data {i} shape: ", df.shape)

    # 调用绘制帕累托前沿的函数
    plot_pareto_front_all_data(all_metric_data_list, all_metric_name_list)