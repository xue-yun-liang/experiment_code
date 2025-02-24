import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

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

# 示例数据
data = pd.DataFrame({
    'power': np.random.randint(5, 50, size=20),
    'latency': np.random.randint(50, 200, size=20)
})
pareto_front = pareto_frontier(data)
print(pareto_front)

# 绘图
plt.figure(figsize=(8, 6))
plt.scatter(data['power'], data['latency'], label='Original Data', color='blue', alpha=0.6)
plt.scatter(pareto_front['power'], pareto_front['latency'], label='Pareto Frontier', color='red', marker='D')
plt.plot(pareto_front['power'], pareto_front['latency'], linestyle='--', color='red')
plt.xlabel('Power')
plt.ylabel('Latency')
plt.title('Pareto Frontier Visualization')
plt.legend()
plt.grid()
plt.savefig('pareto_frontier.png')
