import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

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

def euclidean_distance(point1, point2):
    return np.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2 + (point1[2] - point2[2]) ** 2)


def select_and_calculate_distances(pareto_fronts):
    # 合并所有帕累托前沿
    all_points = pd.concat(pareto_fronts, ignore_index=True)
    
    # 选择两个最好的点
    best_points = all_points.nsmallest(2, ['power', 'latency','Area'])
    
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
        distances.append(total_distance)
    
    return distances, best_points

def pareto_frontier(df: pd.DataFrame) -> pd.DataFrame:
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

def plot_pareto_front_all_data_3d_with_subfig(data_list, names_list):
    # 计算每个数据集的帕累托前沿
    pareto_fronts = []
    for data, name in zip(data_list, names_list):
        cur_pareto = pareto_frontier(data)
        pareto_fronts.append(cur_pareto)

    # 计算距离
    d, bes_p = select_and_calculate_distances(pareto_fronts)
    adrs = pd.DataFrame(names_list, columns=['name'])
    adrs['ADRS'] = d
    print("ADRS:", adrs)
    adrs.to_csv(f'./pareto_data/ADRS_{benchmark}_{target}_{metric_str}.csv')

    # 获取 x, y 和 z 标签
    x_label, y_label, z_label = data_list[0].columns  # 假设顺序是 power, latency, Area

    # 合并所有数据并绘制所有数据的 3D 散点图
    combined_data = pd.concat(data_list, ignore_index=True)
    colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
    markers = ['x', '^', '+', 's', 'd', 'o', '*', 'v', '<', '>']

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
    fig = plt.figure(figsize=(16, 9))

    # 第一个子图：默认视角
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.scatter(combined_data[x_label], combined_data[y_label], combined_data[z_label], color='lightgray', s=60,label='all data', alpha=0.01)

    # 绘制每个数据集的帕累托前沿
    for i, (pareto_front, name) in enumerate(zip(pareto_fronts, names_list)):
        color = colors[i % len(colors)]
        marker = markers[i % len(markers)]
        if name == 'crldse':
            ax1.scatter(pareto_front[x_label], pareto_front[y_label], pareto_front[z_label], color=color, s=60,marker=marker, zorder=10, label=name)
        else:
            ax1.scatter(pareto_front[x_label], pareto_front[y_label], pareto_front[z_label], color=color, s=60,marker=marker, label=name)

    ax1.set_xlim(x_min, x_max)
    ax1.set_ylim(y_min, y_max)
    ax1.set_zlim(z_min, z_max)
    ax1.set_xlabel(x_label)
    ax1.set_ylabel(y_label)
    ax1.set_zlabel(z_label)
    ax1.set_title("(a)", loc='center', pad=20)
    ax1.view_init(elev=10, azim=120)
    ax1.legend()

    # 第二个子图：自定义的不同坐标范围
    ax2 = fig.add_subplot(122, projection='3d')
    #ax2.scatter(combined_data[x_label], combined_data[y_label], combined_data[z_label], color='lightgray', label='all data', alpha=0.01)

    # 绘制每个数据集的帕累托前沿
    for i, (pareto_front, name) in enumerate(zip(pareto_fronts, names_list)):
        color = colors[i % len(colors)]
        marker = markers[i % len(markers)]
        ax2.scatter(pareto_front[x_label], pareto_front[y_label], pareto_front[z_label], color=color, s=60,marker=marker, label=name)

    # 设置不同的 xlim, ylim, zlim
    ax2.set_xlim(0, 5)  # Example of different xlim
    ax2.set_ylim(0, 0.0002)  # Example of different ylim
    ax2.set_zlim(0, 50)  # Example of different zlim

    ax2.set_xlabel(x_label)
    ax2.set_ylabel(y_label)
    ax2.set_zlabel(z_label)
    ax2.set_title("(b)", loc='center', pad=20)
    ax2.view_init(elev=10, azim=120)  # Change the view angle for the second plot
    ax2.legend()

    # 保存图像
    plt.tight_layout()  # To avoid overlap between subplots
    fig_path = f'./pareto_data/pareto_3d_with_custom_limits_{benchmark}_{target}_{metric_str}.png'
    print("saving", fig_path)
    plt.savefig(fig_path, format='png', dpi=600)




if __name__ == "__main__":
    # load data
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
    names_ = ['crldse', 'erdse', 'momprdse', 'acdse', 'nsga2', 'cc-aco', 'ppo', 'dtl']

    # 筛选数据
    all_data = [df[metric_list] for df in dataframes]
    all_metric_name_list = names_
    all_metric_data_list = []
    for data, n in zip(all_data,names):
        print(f"====== current data {n} ======")
        print(data.info())
        filtered_df = filter_invalid(data)
        all_metric_data_list.append(filtered_df)

    # 打印每个筛选后 DataFrame 的形状
    for i, df in enumerate(all_metric_data_list):
        print(f"data {i} shape: ", df.shape)

    # 调用绘制帕累托前沿的函数
    plot_pareto_front_all_data_3d_with_subfig(all_metric_data_list, all_metric_name_list)