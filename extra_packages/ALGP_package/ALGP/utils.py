import numpy as np


def pop_selected_list(
    original_array,
    index_list      
):    
    selected_array = original_array[index_list]
    
    mask = np.ones(len(original_array), dtype=bool)
    
    mask[index_list] = False
    remaining_array = original_array[mask]

    return {
        'selected': selected_array,
        'remaining': remaining_array
    }
