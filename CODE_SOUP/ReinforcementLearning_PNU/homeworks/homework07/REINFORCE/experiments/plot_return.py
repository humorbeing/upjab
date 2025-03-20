import numpy as np
import scipy


path_list = [
    ['logs_save1/m0d1_return.npy','64-32 Normalize'],
    # ['logs_save1/m1d1_return.npy','128-64-32'],
    # ['logs_save1/m2d1_return.npy','32-16'],
    ['logs_save1/m0d0_return.npy','64-32 No Normalize'],
    
]

ppp = [
    ['old_logs/m0r0_return.npy','64-32 One single Run'],
]

np_list = []

for i in path_list:
    np_list.append(np.load(i[0]))

npp_list = []

for i in ppp:
    npp_list.append(np.load(i[0]))

cut = 985


mean_list = []
for i in np_list:
    mean_list.append(np.mean(i, 0))


x_cut = np.arange(1, cut+1)
x = np.arange(1, 500+1)


def smooth(y, box_pts):
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth



smooth_list = []
for i in mean_list:
    smooth_list.append(smooth(i,15)[:cut])


import matplotlib.pyplot as plt




for i, s in enumerate(mean_list):
    plt.plot(x, s, label=path_list[i][1])


# for i, s in enumerate(npp_list):
#     plt.plot(x, s, label=ppp[i][1])

plt.legend()

plt.title('Average Episodic Return')
# plt.title('Episodic Return: 19 vs One')
plt.xlabel('Episode')
plt.ylabel('Return')
plt.xlim(-10,225)
plt.show()

print('end')