

def get_value_list(fix_axis, log_array):
    value_set = set()
    for v in log_array[:,fix_axis]:
        value_set.add(v)
    # print(value_set)
    value_list = list(value_set)
    value_list.sort()
    return value_list
