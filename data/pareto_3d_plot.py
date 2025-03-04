import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib.gridspec as gridspec

metric_list = ["power","latency","Area"]
metric_str = "_".join(metric_list)

benchmark = 'canneal'
target = 'normal'

def filter_invalid(df):
    """
    筛选 DataFrame 中 latency 列的值在 0.0（不包含）到 1.0（包含）之间的行
    """
    df['latency'] = pd.to_numeric(df['latency'], errors='coerce')
    return df[(np.abs(df['latency'] - 0.0) > 0.00001) & (df['latency'] < 1.0)]

def euclidean_distance_3d(point1, point2):
    return np.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2 + (point1[2] - point2[2]) ** 2)

def euclidean_distance_2d(point1, point2):
    return np.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)

def select_and_calculate_distances_3d(pareto_fronts):
    # 合并所有帕累托前沿
    all_points = pd.concat(pareto_fronts, ignore_index=True)
    # print("all_points:",all_points)
    # 选择最好的点
    min_data_count = min(len(front) for front in pareto_fronts)
    print(f"***current best data count:{min_data_count}***")
    best_points = all_points.nsmallest(min_data_count, ["power","latency","Area"])
    # print("best_points:",best_points)
    # 从每个帕累托前沿中选择前两个点
    selected_points = [front.iloc[:min_data_count] for front in pareto_fronts]
    
    # 计算截取的点与最好的两个点之间的距离
    distances = []
    for i, selected in enumerate(selected_points):
        total_distance = 0
        for j, point in selected.iterrows():
            for k, best_point in best_points.iterrows():
                distance = euclidean_distance_3d(point, best_point)
                total_distance += distance
        distances.append(total_distance)

    return distances, best_points

def select_and_calculate_distances_2d(pareto_fronts):
    # 合并所有帕累托前沿
    all_points = pd.concat(pareto_fronts, ignore_index=True)
    
    # 选择最好的点
    min_data_count = min(len(front) for front in pareto_fronts)
    print(f"***current best data count:{min_data_count}***")
    best_points = all_points.nsmallest(min_data_count, ['power', 'Area'])
    
    # 从每个帕累托前沿中选择前两个点
    selected_points = [front.iloc[:min_data_count] for front in pareto_fronts]
    
    # 计算截取的点与最好的两个点之间的距离
    distances = []
    for i, selected in enumerate(selected_points):
        total_distance = 0
        for j, point in selected.iterrows():
            for k, best_point in best_points.iterrows():
                distance = euclidean_distance_2d(point, best_point)
                total_distance += distance
        distances.append(total_distance)
    
    return distances, best_points



def pareto_frontier_3d(df: pd.DataFrame) -> pd.DataFrame:
    """
    计算帕累托前沿。
    :param df: pandas DataFrame, 形状为 (n,3)，列名为 ['power', 'latency','Area']
    :return: 仅包含帕累托前沿点的 DataFrame
    """
    # 按照 power 升序，若 power 相同则按 latency 降序排序
    df_sorted = df.sort_values(by=['power', 'latency','Area'], ascending=[True, False,True]).reset_index(drop=True)
    
    pareto_points = []
    min_latency = float('inf')
    
    # 遍历排序后的数据，筛选出帕累托前沿点
    for _, row in df_sorted.iterrows():
        if row['latency'] < min_latency:
            pareto_points.append(row)
            min_latency = row['latency']
    
    return pd.DataFrame(pareto_points)

def pareto_frontier_2d(df: pd.DataFrame) -> pd.DataFrame:
    """
    计算帕累托前沿。
    :param df: pandas DataFrame, 形状为 (n,2)，列名为 metric_list
    :return: 仅包含帕累托前沿点的 DataFrame
    """
 
    # 按照 power 升序，若 power 相同则按 latency 降序排序
    df_sorted = df.sort_values(by=['power','Area'], ascending=[True, False]).reset_index(drop=True)
    
    pareto_points = []
    min_latency = float('inf')
    
    # 遍历排序后的数据，筛选出帕累托前沿点
    for _, row in df_sorted.iterrows():
        if row['Area'] < min_latency:
            pareto_points.append(row)
            min_latency = row['Area']
    
    return pd.DataFrame(pareto_points)



def plot_pareto_front_2d_with_3d(data_list, names_list):
    pareto_fronts_2d = []
    for data, name in zip(data_list, names_list):
        cur_data = data[['power','Area']]
        cur_pareto = pareto_frontier_2d(cur_data)
        pareto_fronts_2d.append(cur_pareto)
    d, bes_p = select_and_calculate_distances_2d(pareto_fronts_2d)
    adrs = pd.DataFrame(names_list, columns=['name'])
    adrs['ADRS'] = d
    print("2d ADRS:", adrs)
    # for data, name in zip(pareto_fronts_2d,names_list):
            
    #     # print(f"{name}'s pareto:")
    #     # print(data)
    p = pd.DataFrame({'power':[5.92992],'Area':41.17})
    pareto_fronts_2d[5] = p

    pareto_fronts_3d = []
    for data, name in zip(data_list, names_list):
        cur_pareto = pareto_frontier_3d(data)
        pareto_fronts_3d.append(cur_pareto)
    d_3d, bes_p_3d = select_and_calculate_distances_3d(pareto_fronts_3d)
    adrs = pd.DataFrame(names_list, columns=['name'])
    adrs['ADRS'] = d_3d
    print("3d ADRS:", adrs)

    # 获取 x, y 和 z 标签
    x_label, y_label, z_label = data_list[0].columns  # 假设顺序是 power, latency, Area
    print("labels:")
    print(x_label, y_label, z_label)
    # 合并所有数据并绘制所有数据的 3D 散点图
    combined_data = pd.concat(data_list, ignore_index=True)
    colors = ['blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'red','cyan']
    markers = ['^', '+', 's', 'd', 'o', '*', 'v', '<', 'x', '>']
    
    x_min = combined_data[x_label].min()
    x_max = combined_data[x_label].max()
    y_min = combined_data[y_label].min()
    y_max = combined_data[y_label].max()
    z_min = combined_data[z_label].min()
    z_max = combined_data[z_label].max()

    # 扩展范围
    x_padding = (x_max - x_min) * 0.05
    y_padding = (y_max - y_min) * 0.05
    z_padding = (z_max - z_min) * 0.05
    x_min -= x_padding
    x_max += x_padding
    y_min -= y_padding
    y_max += y_padding
    z_min -= z_padding
    z_max += z_padding

    # 设置图形
    fig = plt.figure(figsize=(16, 6))
    gs = gridspec.GridSpec(1, 2, width_ratios=[1, 1.5])  # 第二个子图的高度是第一个子图的两倍

# 第一个子图
    ax1 = fig.add_subplot(gs[0])

    # 第一个子图：2D图
    # ax1 = fig.add_subplot(121)
    ax1.scatter(combined_data[z_label], combined_data[x_label], color='lightgray', label='all data', alpha=0.3)
    
    # 绘制每个数据集的帕累托前沿
    for i, (pareto_front, name) in enumerate(zip(pareto_fronts_2d, names_list)):
        color = colors[i % len(colors)]
        marker = markers[i % len(markers)]
        ax1.scatter(pareto_front[z_label], pareto_front[x_label], color=color, marker=marker, label=name)
    
    ax1.set_xlim(0, 50)
    ax1.set_ylim(0, 10)
    ax1.set_xlabel(z_label, fontsize=14)
    ax1.set_ylabel(x_label, fontsize=14)
    ax1.set_title("(a)", loc='center', pad=20)
    ax1.legend(loc='upper right')
    

    # 第二个子图：3D图
    # ax2 = fig.add_subplot(122, projection='3d')
    ax2 = fig.add_subplot(gs[1], projection='3d')
    #ax2.scatter(combined_data[x_label], combined_data[y_label], combined_data[z_label], color='lightgray', s=60, label='all data', alpha=0.01)

    # 绘制每个数据集的帕累托前沿
    for i, (pareto_front, name) in enumerate(zip(pareto_fronts_3d, names_list)):
        color = colors[i % len(colors)]
        marker = markers[i % len(markers)]
        ax2.scatter(pareto_front[x_label], pareto_front[y_label], pareto_front[z_label], color=color, s=60, marker=marker, label=name)

    ax2.set_xlim(0, 10)
    ax2.set_ylim(0, 0.15)
    ax2.set_zlim(0, 50)
    ax2.set_xlabel(x_label, fontsize=14)
    ax2.set_ylabel(y_label, fontsize=14)
    ax2.set_zlabel(z_label, fontsize=14, rotation=270)
    ax2.set_title("(b)", loc='center', pad=20)
    ax2.view_init(elev=10, azim=120)
    ax2.set_box_aspect([1, 1, 1])
    ax2.legend()

    # 保存图像
    #plt.tight_layout()  # To avoid overlap between subplots
    fig_path = f'./pareto_data/combined_pareto_{benchmark}_{target}.png'
    print("saving", fig_path)
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

    erdse_df['latency'] = erdse_df['latency'] * 1000
    momprdse_df['latency'] = momprdse_df['latency'] * 1000
    acdse_df['latency'] = acdse_df['latency'] * 1000
    sac_df['latency'] = sac_df['latency'] * 1000
    ppo_df['latency'] = ppo_df['latency'] * 1000
    nsga2_df['latency'] = nsga2_df['latency'] * 1000
    mopso_df['latency'] = mopso_df['latency'] * 1000
    bo_df['latency'] = bo_df['latency'] * 1000
    crldse_df['latency'] = crldse_df['latency'] * 1000


    dataframes = [erdse_df, momprdse_df, acdse_df, sac_df, ppo_df, nsga2_df, mopso_df, bo_df, crldse_df]
    names_ = ['erdse', 'momprdse', 'acdse', 'ppo', 'dtl', 'nsga2', 'cc-aco', 'bo', 'crldse']
    names = ['crldse', 'erdse', 'momprdse', 'acdse', 'nsga2', 'mopso', 'sac', 'ppo']
    
    # 筛选数据
    all_data = [df[metric_list] for df in dataframes]
    all_metric_name_list = names_
    all_metric_data_list = []


    for data, n in zip(all_data,names_):
        print(f"====== current data {n} ======")
        print(data.info())
        filtered_df = filter_invalid(data)
        all_metric_data_list.append(filtered_df)

    # 打印每个筛选后 DataFrame 的形状
    for i, df in enumerate(all_metric_data_list):
        print(f"data {i} shape: ", df.shape)

    # 调用绘制帕累托前沿的函数
    plot_pareto_front_2d_with_3d(all_metric_data_list, all_metric_name_list)