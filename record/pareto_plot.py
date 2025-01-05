import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# load data
metric = pd.read_csv('ri_metric.csv')

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

def plot_pareto_front(data, pareto_front):
    x_label, y_label = data.columns[:2]
    x_label, y_label = y_label, x_label
    plt.scatter(data[x_label], data[y_label], color='lightgray', label='no pareto')
    plt.scatter(pareto_front[x_label], pareto_front[y_label], color='red', marker='x', label='pareto')
    plt.legend()
    
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.savefig('pareto.png',format='png')

if __name__ == "__main__":
    power_latency = metric[["power","latency"]]
    power_latency = power_latency[power_latency['latency']!= 0]
    cur_pareto_front = pareto_front(power_latency)
    plot_pareto_front(power_latency,cur_pareto_front)