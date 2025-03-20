
import matplotlib.pyplot as plt
import json

with open('logs/lr.json', 'r') as f:
    data_log = json.load(f)

# with open('logs/gamma.json', 'r') as f:
#     data_log = json.load(f)


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
x = list(range(1,17))
# plt.scatter(x, gamma_list)
# plt.scatter(gamma_list, duration_list)
plt.scatter(x, duration_list)
# for i in range(len(x)): 
#     plt.annotate(str(gamma_list[i]), (x[i], gamma_list[i] + 0.2)) 

mean_list = []
for i in np_list:
    mean_list.append(np.mean(i, 0))

x = np.arange(1, 1000+1)


# for i, s in enumerate(mean_list):
#     # plt.plot(x, s, label=label_list[i])
#     plt.plot(x, s)

# plt.legend()
plt.title('Time to complete')
plt.xlabel('Experiment Number')
plt.ylabel('Second')
plt.ylim(150,200)
plt.show()

print('end')