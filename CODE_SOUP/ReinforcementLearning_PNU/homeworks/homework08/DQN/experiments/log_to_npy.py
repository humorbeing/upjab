save_name = 'v08'
file_path = f'logs/{save_name}_ok.log'


import numpy as np
import matplotlib.pyplot as plt
with open(file_path, 'r') as f:
    _log = f.readlines()





def get_n(s):
    # score = s.split('|')[-2]
    ep = s.split('|')[3]
    sc = s.split('|')[5]
    ep = int(ep)
    sc = float(sc)
    
    return ep, sc

ss = []
current = None
last_score = None
for lg in _log:
    ep, score = get_n(lg)
    if current is None:
        current = ep
    
    if current == ep:
        last_score = score
    else:
        ss.append(last_score)
        current = ep
        last_score = score
    

ss.append(last_score)

sss = np.array(ss)
np.save(f'logs/{save_name}.npy', sss)
plt.plot(list(range(len(sss))), ss)
# plt.xlabel('original episodic return')
# plt.title('Histogram of episodic return of 102 random run ')
plt.show()
print('')

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