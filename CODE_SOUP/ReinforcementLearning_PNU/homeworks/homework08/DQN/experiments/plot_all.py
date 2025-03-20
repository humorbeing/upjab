
path_list = [
    
    
    # ['logs/v01.npy', 'R=|x|+v^2, Replay=500'],
    # ['logs/random.npy', 'Random: Baseline'],
    # ['logs/v06.npy', 'R= x^2 + v^2, Replay=2000'],
    # ['logs/v07.npy', 'R=|x|, replay=2000'],
    # ['logs/v02.npy', 'R=|x|+|v|, Replay=500'], 
    # ['logs/v03.npy', 'R=|x|+|v|, Replay=50000'],
    # ['logs/v04.npy', 'R=|x|+|v|, Replay=2000'], 
    # ['logs/v05.npy', 'R= x * v, Replay=500'],
    
    
    ['logs/v08.npy', 'R=2 * x * v, Replay=2000'], 
    # ['logs/v09.npy', 'R= 2*(x+v)^2, xv>0, Replay=2000'],
    # ['logs/v10.npy', 'R=(x + v)^2, Replay=2000'], 
    # ['logs/.npy', ''],
    # ['logs/.npy', ''],
    # ['logs/.npy', ''],
    # ['logs/.npy', ''], 
    # ['logs/.npy', ''],
    # ['logs/.npy', ''], 
    # ['logs/.npy', ''],
    # ['logs/.npy', ''],    
]


import numpy as np
import scipy

np_list = []

for i in path_list:
    np_list.append(np.load(i[0]))



# x = list(range(1,1000+1))
import matplotlib.pyplot as plt



for i, s in enumerate(np_list):
    plt.plot(list(range(1, len(s)+1)), s, label=path_list[i][1])
# plt.plot(x, e_mean, label='MC every visit')
# plt.plot(x, f_mean, label='MC first visit')
# plt.plot(x, t1_mean, label='e10')
# plt.plot(x, t3_mean, label='f1')
# plt.plot(x, t5_mean, label='f2')
# plt.plot(x, t7_mean, label='f3')
# plt.plot(x, f4_mean, label='f4')
# plt.plot(x, f5_mean, label='f5')


# plt.plot(x, e_std, label='MC every visit')
# plt.plot(x, f_std, label='MC first visit')
# plt.plot(x, t_std, label='TD')


# plt.plot(x, e_skew, label='MC every visit')
# plt.plot(x, f_skew, label='MC first visit')
# plt.plot(x, t_skew, label='TD')



# plt.plot(x, e_mode, label='MC every visit')
# plt.plot(x, f_mode, label='MC first visit')
# plt.plot(x, t_mode, label='TD')
plt.legend()
plt.title('MountainCar-v0: Undiscounted Reward Sum of One Run (Not average of Many)')
# plt.ylim((0,1))
plt.xlabel('Episode')
plt.ylabel('Undiscounted Reward Sum')
plt.show()

print('end')