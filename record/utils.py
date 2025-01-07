import numpy as np

def make_non_decreasing(arr):
    new_arr = [arr[0]]  
    for i in range(1, len(arr)):
        new_arr.append(max(new_arr[-1], arr[i]))
    
    return new_arr


def moving_average(a, window_size):
    cumulative_sum = np.cumsum(np.insert(a, 0, 0)) 
    middle = (cumulative_sum[window_size:] - cumulative_sum[:-window_size]) / window_size
    r = np.arange(1, window_size-1, 2)
    begin = np.cumsum(a[:window_size-1])[::2] / r
    end = (np.cumsum(a[:-window_size:-1])[::2] / r)[::-1]
    return np.concatenate((begin, middle, end))

def compute_square(data1, data2, data3, ddl):
    data1 = data1[:ddl]
    data2 = data2[:ddl]
    data3 = data3[:ddl]

    return [sum(data1),sum(data2),sum(data3)]