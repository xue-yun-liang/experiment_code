import matplotlib.pyplot as plt

# 数据
process_nums = [1, 2, 3, 4, 5, 6, 7, 8]
times = [1077.19, 1050.15, 1202.18, 1095.05, 1145.36, 1154.23, 1226.12, 1203.94]
speedups = [1077.19 / time for time in times]
for i in range(len(speedups)):
    speedups[i] = speedups[i] * (i+1)
ideal_speedups = process_nums

# 绘制折线图
plt.plot(process_nums, speedups, marker='o', label='actual')
plt.plot(process_nums, ideal_speedups, marker='x', linestyle='--', label='ideal')

plt.xlabel('num of process')
plt.ylabel('speedup')
plt.grid(True)
plt.savefig('speedup.png',dpi=600)