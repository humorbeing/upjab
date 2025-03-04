
save_name = 'm0d0'
file_path = f'logs_save1/{save_name}.log'




with open(file_path, 'r') as f:
    _log = f.readlines()


_log = _log[:-1]
import numpy as np
import scipy

def get_n(s):
    _, _round, _, _episode, _, _on_return, _, _on_len, _, _global_step, _ = s.split('|')
    round_ = int(_round)
    episode_ = int(_episode)
    on_return = float(_on_return)
    on_len = int(_on_len)
    global_step = float(_global_step)    
    return round_, episode_, on_return, on_len, global_step


len(_log) / 500

assert len(_log) % 500 == 0

rou = int(len(_log) / 500)

return_matrix = []
len_matrix = []

for r in range(rou):
    
    return_list = []
    len_list = []
    for ep in range(500):        
        one_log = _log[r*500 + ep]
        round_, episode_, on_return, on_len, global_step = get_n(one_log)
        assert round_ == r + 1
        assert ep == episode_
        return_list.append(on_return)
        len_list.append(on_len)
        
    return_matrix.append(return_list)
    len_matrix.append(len_list)
   





import numpy as np

on_return_np = np.array(return_matrix)
on_len_np = np.array(len_matrix)




np.save(f'logs/{save_name}_return.npy', on_return_np)
np.save(f'logs/{save_name}_len.npy', on_len_np)



print("end")