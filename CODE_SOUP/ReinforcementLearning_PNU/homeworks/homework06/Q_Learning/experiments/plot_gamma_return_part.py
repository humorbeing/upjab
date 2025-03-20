
import matplotlib.pyplot as plt
import json

# with open('logs/lr.json', 'r') as f:
#     data_log = json.load(f)

with open('logs/gamma.json', 'r') as f:
    data_log = json.load(f)


import numpy as np


np_list = []
gamma_list = []
lr_list = []
episode_len_list = []
duration_list = []
for dd in data_log:
    temp11 = data_log[dd]
    temp12 = temp11['return']
    temp13 = np.array(temp12)
    np_list.append(temp13)

    temp21 = temp11['alpha']
    lr_list.append(temp21)
    gamma_list.append(temp11['gamma'])
    duration_list.append(temp11['duration'])
    temp31 = temp11['length']
    temp32 = np.array(temp31)
    episode_len_list.append(temp32)


label_list = gamma_list



mean_list = []
for i in np_list:
    mean_list.append(np.mean(i, 0))

x = np.arange(1, 1000+1)

# in_here = [22,21,20,19,18]
in_here = [0,1,2,3,4,5]
# in_here = [6,7,8,9,10,11]
# in_here = [12,13,14,15,16,17]
in_here = [
    # 0,
    # 1,2,3,4,5,
    # 6,7,8,9,10,11,
    # 12,13,14,15,16,17,
    18,
    # 19,
    20,21,22
    ]
for i, s in enumerate(mean_list):
    if i in in_here:
        plt.plot(x, s, label=label_list[i])
    # plt.plot(x, s)

plt.legend()
plt.title('Different discount_factor experiment: 19,21,22,23')
plt.xlabel('Episode')
plt.ylabel('Average Episodic Return')

plt.show()

print('end')