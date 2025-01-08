import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

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

def pareto_front(df):
    pareto_front = []
    
    for index, row in df.iterrows():
        is_pareto = True

        for other_index, other_row in df.iterrows():
            if other_index != index and (other_row <= row).all():
                is_pareto = False
                break

        if is_pareto:
            pareto_front.append(row)

    pareto_front_df = pd.DataFrame(pareto_front)
    
    return pareto_front_df

def plot_pareto_front(data1, data2, data3,names):
    pareto_front1 = pareto_front(data1)
    pareto_front2 = pareto_front(data2)
    pareto_front3 = pareto_front(data3)
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

if __name__ == "__main__":
    # load data
    crldse_metric_df = pd.read_csv('../data/canneal_normal_crldse.csv')
    erdse_metric_df = pd.read_csv('../data/canneal_normal_erdse.csv')
    momprdse_metric_df = pd.read_csv('../data/canneal_normal_momprdse.csv')
    crldse_power_latency = crldse_metric_df[["power","latency"]]
    erdse_power_latency = erdse_metric_df[["power","latency"]]
    momprdse_power_latency = momprdse_metric_df[["power","latency"]]
    crldse_power_latency = crldse_power_latency[crldse_power_latency['latency']!= 0]
    erdse_power_latency = erdse_power_latency[erdse_power_latency['latency']!=0]
    momprdse_power_latency = momprdse_power_latency[momprdse_power_latency['latency']!=0]
    plot_pareto_front(crldse_power_latency,erdse_power_latency,momprdse_power_latency,['crldse','erdse','momprdse'])