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

import pandas as pd

def compute_square(data_lists, names_list, ddl):
    # 检查 names_list 的长度是否与 data_lists 的数量一致
    if len(names_list) != len(data_lists):
        raise ValueError("The length of names_list must match the number of data_lists.")
    
    results = []
    
    for data in data_lists:
        # 截取前 ddl 个元素
        truncated_data = data[:ddl]
        results.append(sum(truncated_data))
    
    df = pd.DataFrame({'Name': names_list, 'Sum': results})
    
    return df


def shift_fill(data_list, shift_amount):
    n = len(data_list)
    shift_amount = shift_amount % n
    shifted_list = data_list[shift_amount:] + [data_list[-1]] * shift_amount
    return shifted_list

def find_first_decreasing_position(lst):
    for i in range(len(lst) - 1, 0, -1):
        if lst[i] < lst[i - 1]:
            return i
    return -1

def find_all_decreasing_positions(data_list,name_list):
    positions = []
    for i in range(len(data_list)):
        positions.append(find_first_decreasing_position(data_list[i]))
    res = pd.DataFrame({'Name': name_list, 'Position': positions})
    return res

def filter_invalid(df):
    # 将大于100的数置为0
    df[df > 100] = 0
    return df
