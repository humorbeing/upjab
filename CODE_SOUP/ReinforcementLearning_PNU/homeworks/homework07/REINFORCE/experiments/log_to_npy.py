
file_path = 'logs/m2r0.log'
save_name = 'm2r0'



with open(file_path, 'r') as f:
    _log = f.readlines()



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

return_list = []

for one_log in _log:
    round_, episode_, on_return, on_len, global_step = get_n(one_log)
    return_list.append(on_return)
   





import numpy as np

on_return_np = np.array(return_list)





np.save(f'logs/{save_name}_return.npy', on_return_np)



print("end")