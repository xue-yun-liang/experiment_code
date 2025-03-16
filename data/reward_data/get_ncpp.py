
def process_lists(list1, list2):
    if len(list1) != len(list2):
        raise ValueError("两个列表的长度必须相同")
    
    max1 = max(list1)
    max2 = max(list2)
    
    result = []
    for i in range(len(list1)):
        normalized1 = list1[i] / max1
        normalized2 = list2[i] / max2
        result.append(normalized1 * normalized2)
    
    return result

if __name__ == "__main__":
    list1 = [1.7021, 0.2877, 0.5810, 0.5228, 1.6008, 2.2303, 0.4424, 1.6942, 0.2957]    # TC列
    list2 =  [244, 232, 131, 248, 163, 290, 127, 72, 98]  # ADRS列
    print(process_lists(list1, list2))
