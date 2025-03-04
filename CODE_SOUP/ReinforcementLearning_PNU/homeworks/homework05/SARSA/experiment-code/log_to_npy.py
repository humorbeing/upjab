
file_path = 'logs/e01.log'
save_name = 'e01'



with open(file_path, 'r') as f:
    _log = f.readlines()


# log_ = _log[1:]

def get_n(s):
    _, _round, _, _episode, _, _on_return, _, _on_len, _, _epsilon, _, _evl_return, _, _evl_len, _ = s.split('|')
    round_ = int(_round)
    episode_ = int(_episode)
    on_return = float(_on_return)
    on_len = int(_on_len)
    epsilon = float(_epsilon)
    evl_return = float(_evl_return)
    evl_len = int(_evl_len)
    return round_, episode_, on_return, on_len, epsilon, evl_return, evl_len

on_return_matrix = []
on_length_matrix = []
epsilon_matrix = []
eval_return_matrix = []
eval_length_matrix = []


for r in range(999):
    on_return_list = []
    on_length_list = []
    epsilon_list = []
    eval_return_list = []
    eval_length_list = []
    for episode in range(1000):
        index = r * 1000 + episode
        ep_log = _log[index]
        
        round_, episode_, on_return, on_len, epsilon, evl_return, evl_len = get_n(ep_log)
        # assert round_ == r
        assert episode_ == episode
        

        on_return_list.append(on_return)
        on_length_list.append(on_len)
        epsilon_list.append(epsilon)
        eval_return_list.append(evl_return)
        eval_length_list.append(evl_len)     
        print(f'{r}-{episode}')
    
    on_return_matrix.append(on_return_list)
    on_length_matrix.append(on_length_list)
    epsilon_matrix.append(epsilon_list)
    eval_return_matrix.append(eval_return_list)
    eval_length_matrix.append(eval_length_list)


import numpy as np

on_return_np = np.array(on_return_matrix)
on_length_np = np.array(on_length_matrix)
epsilon_np = np.array(epsilon_matrix)
eval_return_np = np.array(eval_return_matrix)
eval_length_np = np.array(eval_length_matrix)




np.save(f'logs/{save_name}_on_return.npy', on_return_np)
np.save(f'logs/{save_name}_on_length.npy', on_length_np)
np.save(f'logs/{save_name}_epsilon.npy', epsilon_np)
np.save(f'logs/{save_name}_eval_return.npy', eval_return_np)
np.save(f'logs/{save_name}_eval_length.npy', eval_length_np)


print("end")