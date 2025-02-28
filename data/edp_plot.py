import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

np.set_printoptions(suppress=False, precision=32, floatmode='fixed')
benchmark = 'canneal'
target = 'cloud'

# 定义数据框架
dataframes = []

def load_and_process_data(filename):
    df = pd.read_csv(filename,index_col=0)
    df['edp'] = df['latency'] * df['energy']
    return df

def filter_invalid(df):
    """
    筛选 DataFrame 中 latency 列的值在 0.0（不包含）到 1.0（包含）之间的行
    """
    df['latency'] = pd.to_numeric(df['latency'], errors='coerce')
    df = df.dropna(subset=['latency'])
    return df[(np.abs(df['latency'] - 0.0) > 0.00001) & (df['latency'] < 999.0)]



def make_non_decreasing(arr):
    new_arr = [arr[0]]  
    for i in range(1, len(arr)):
        new_arr.append(max(new_arr[-1], arr[i]))
    
    return new_arr

def make_non_increasing(arr):
    new_arr = [arr[0]]  
    for i in range(1, len(arr)):
        new_arr.append(min(new_arr[-1], arr[i]))

    return new_arr

if __name__ == '__main__':
    # 加载数据
    crldse_df = load_and_process_data(f'{benchmark}_{target}_crldse.csv')
    erdse_df = load_and_process_data(f'{benchmark}_{target}_erdse.csv')
    momprdse_df = load_and_process_data(f'{benchmark}_{target}_momprdse.csv')
    acdse_df = load_and_process_data(f'{benchmark}_{target}_acdse.csv')
    nsga2_df = load_and_process_data(f'{benchmark}_{target}_nsga2.csv')
    #mopso_df = load_and_process_data(f'{benchmark}_{target}_mopso.csv')
    sac_df = load_and_process_data(f'{benchmark}_{target}_sac.csv')
    ppo_df = load_and_process_data(f'{benchmark}_{target}_ppo.csv')
    erdse_df = erdse_df.dropna()
    dataframes = [
        ('crldse_df', crldse_df),
        ('erdse_df', erdse_df),
        ('momprdse_df', momprdse_df),
        ('acdse_df', acdse_df),
        ('nsga2_df', nsga2_df),
#        ('mopso_df', mopso_df),
        ('sac_df', sac_df),
        ('ppo_df', ppo_df)
    ]

    for name, df in dataframes:
        df[:] = filter_invalid(df)


    # 绘制单调递减的edp曲线
    plt.figure(figsize=(12, 8))

    for name, df in dataframes:
        # 找到单调递减的edp值
        df['monotonic_edp'] = make_non_increasing(df['edp'])
        print(f"{name}: {df['edp'].min()}")

        # 绘制曲线
        if name == 'crldse_df':
            plt.plot(df.index, df['monotonic_edp'], label=name, color='red',zorder=10)
        elif name == 'nsga2_df':
            plt.plot(df.index, df['monotonic_edp'], label=name, color='blue',zorder=9)
        else:
            plt.plot(df.index, df['monotonic_edp'], label=name,)

    # 设置图表标题和图例
    plt.ylim(0,0.002)
    plt.xlim(0,20)
    plt.xlabel('Index')
    plt.ylabel('EDP')
    plt.legend()
    plt.grid(True)
    print(f"saving edp_data/{benchmark}_{target}_edp_with_limit.png")
    plt.savefig(f'edp_data/{benchmark}_{target}_edp_with_limit.png')