
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
    list1 =  [204, 193, 185, 165, 68, 230, 320, 95, 98]  # TC列
    list2 =  [40.0899, 0.3134, 24.1808, 25.2503, 65.3623, 5.8150, 6.4893, 15.1719, 2.4389]  # ADRS列
    print(process_lists(list1, list2))