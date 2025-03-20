
file_path = 'logs/gamma.log'
save_name = 'lr'



with open(file_path, 'r') as f:
    _log = f.readlines()


# log_ = _log[1:]

def get_n(s):
    _, _round, _, _episode, _, _on_return, _, _on_len, _, _alpha, _, _gamma, _ = s.split('|')
    round_ = int(_round)
    episode_ = int(_episode)
    on_return = float(_on_return)
    on_len = int(_on_len)
    alpha = float(_alpha)
    gamma = float(_gamma)    
    return round_, episode_, on_return, on_len, alpha, gamma

def get_end(s):
    _, duration_, _, alpha_, _, gamma_, _ = s.split('|')
    duration = float(duration_)
    alpha = float(alpha_)
    gamma = float(gamma_)
    return duration, alpha, gamma
    # print('')

import numpy as np
on_return_matrix = np.zeros((999, 1000)) - 999
on_length_matrix = np.zeros((999, 1000)) - 999
log_data = {}

counter = 0
for lo in _log:
    checker = lo.split('|')[0].strip()[-5:]
    if checker == 'round':
        round_, episode_, on_return, on_len, alpha, gamma = get_n(lo)
        on_return_matrix[round_, episode_] = on_return
        on_length_matrix[round_, episode_] = on_len
    else:
        _duration, _alpha, _gamma = get_end(lo)
        if (alpha == _alpha) and (_gamma == gamma):
            temp_log = {
                'alpha': _alpha,
                'gamma': _gamma,
                'duration': _duration,
                'return': on_return_matrix,
                'length': on_length_matrix
            }
            log_data[counter] = temp_log
            counter = counter + 1
            on_return_matrix = np.zeros((999, 1000)) - 999
            on_length_matrix = np.zeros((999, 1000)) - 999

        else: 
            print('something is wrong.')


import json


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


with open('data.json', 'w') as f:
    json.dump(log_data, f, cls=NumpyEncoder)


print("end")