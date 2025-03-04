
file_path1 = 'logs/f12p1.log'
file_path2 = 'logs/f12p2.log'
file_path3 = 'logs/f11p3.log'
file_path3 = None
save_name = 'f12'

_log = []

with open(file_path1, 'r') as f:
    _log1 = f.readlines()

with open(file_path2, 'r') as f:
    _log2 = f.readlines()

if file_path3:
    with open(file_path3, 'r') as f:
        _log3 = f.readlines()

    _log.extend(_log3)


_log.extend(_log1)
_log.extend(_log2)

# log_ = _log[1:]

def get_n(s):
    _, _round, _, _episode, _, _score, _, _epsilon, _ = s.split('|')
    round_ = int(_round)
    episode = int(_episode)
    score = float(_score)
    epsilon = float(_epsilon)
    return round_, episode, score, epsilon

return_matrix = []
epsilon_matrix = []
ROUND = len(_log) // 1000
for r in range(ROUND):
    retrun_list = []
    epsilon_list = []
    for episode in range(1000):
        index = r * 1000 + episode
        ep_log = _log[index]
        
        round_, episode_, score, epsilon = get_n(ep_log)
        # assert round_ == r
        assert episode_ == episode
        

        retrun_list.append(score)   
        epsilon_list.append(epsilon)     
        print(f'{r}-{episode}')
    
    return_matrix.append(retrun_list)
    epsilon_matrix.append(epsilon_list)


import numpy as np

return_np = np.array(return_matrix)[:999,...]
epsilon_np = np.array(epsilon_matrix)[:999,...]


np.save(f'logs/{save_name}_return.npy', return_np)
np.save(f'logs/{save_name}_epsilon.npy', epsilon_np)


print("end")