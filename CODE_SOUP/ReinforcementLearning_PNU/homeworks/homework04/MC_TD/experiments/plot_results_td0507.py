td01_path = 'logs/td01.npy'
td03_path = 'logs/td03.npy'
td05_path = 'logs/td05.npy'
td07_path = 'logs/td07.npy'
mc_every_path = 'logs/every.npy'
mc_first_path = 'logs/first.npy'

import numpy as np
import scipy

tt1 = np.load(td01_path)
tt3 = np.load(td03_path)
tt5 = np.load(td05_path)
tt7 = np.load(td07_path)
ee = np.load(mc_every_path)
ff = np.load(mc_first_path)

cut = 350

# t_mode = scipy.stats.mode(tt, 0)[0][0]
# e_mode = scipy.stats.mode(ee, 0)[0][0]
# f_mode = scipy.stats.mode(ff, 0)[0][0]


# t_skew = scipy.stats.skew(tt, 0)
# e_skew = scipy.stats.skew(ee, 0)
# f_skew = scipy.stats.skew(ff, 0)


t1_mean = np.mean(tt1, 0)
t3_mean = np.mean(tt3, 0)
t5_mean = np.mean(tt5, 0)
t7_mean = np.mean(tt7, 0)
e_mean = np.mean(ee, 0)
f_mean = np.mean(ff, 0)

# t_mean_cut = np.mean(tt, 0)[:cut]
# e_mean_cut = np.mean(ee, 0)[:cut]
# f_mean_cut = np.mean(ff, 0)[:cut]

# t_median = np.median(tt, 0)
# e_median = np.median(ee, 0)
# f_median = np.median(ff, 0)

# t_25 = np.quantile(tt, 0.35, 0)
# e_25 = np.quantile(ee, 0.35, 0)
# f_25 = np.quantile(ff, 0.35, 0)

# t_75 = np.quantile(tt, 0.65, 0)
# e_75 = np.quantile(ee, 0.65, 0)
# f_75 = np.quantile(ff, 0.65, 0)

# t_05 = np.quantile(tt, 0.15, 0)
# e_05 = np.quantile(ee, 0.15, 0)
# f_05 = np.quantile(ff, 0.15, 0)

# t_95 = np.quantile(tt, 0.85, 0)
# e_95 = np.quantile(ee, 0.85, 0)
# f_95 = np.quantile(ff, 0.85, 0)

# t_25 = np.percentile(tt, 25, 0)
# e_25 = np.percentile(ee, 0.35, 0)
# f_25 = np.percentile(ff, 0.35, 0)

# t_75 = np.percentile(tt, 75, 0)
# e_75 = np.percentile(ee, 0.65, 0)
# f_75 = np.percentile(ff, 0.65, 0)

# t_05 = np.percentile(tt, 5, 0)
# e_05 = np.percentile(ee, 0.15, 0)
# f_05 = np.percentile(ff, 0.15, 0)

# t_95 = np.percentile(tt, 95, 0)
# e_95 = np.percentile(ee, 0.85, 0)
# f_95 = np.percentile(ff, 0.85, 0)

# t_std = np.std(tt, 0)
# e_std = np.std(ee, 0)
# f_std = np.std(ff, 0)

x_cut = np.arange(1, cut+1)
x = np.arange(1, 1000+1)


def smooth(y, box_pts):
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth

# t_1 = smooth(t_mean,3)[:cut]
# t_2 = smooth(t_mean,15)[:cut]
# t_3 = smooth(t_mean,30)[:cut]

t1_2 = smooth(t1_mean,15)[:cut]
t3_2 = smooth(t3_mean,15)[:cut]
t5_2 = smooth(t5_mean,15)[:cut]
t7_2 = smooth(t7_mean,15)[:cut]

e_1 = smooth(e_mean,3)[:cut]
e_2 = smooth(e_mean,15)[:cut]
e_3 = smooth(e_mean,30)[:cut]

f_1 = smooth(f_mean,3)[:cut]
f_2 = smooth(f_mean,15)[:cut]
f_3 = smooth(f_mean,30)[:cut]

import matplotlib.pyplot as plt

# plt.plot(x, t_mean)
# plt.plot(x_cut, t_1)
# plt.plot(x_cut, t_2)
# plt.plot(x_cut, t_3)

# plt.plot(x_cut, e_mean_cut, label='MC every visit')
# plt.plot(x_cut, f_mean_cut, label='MC first visit')
# plt.plot(x_cut, t_mean_cut, label='TD')

# plt.plot(x_cut, e_2, label='MC every visit')
# plt.plot(x_cut, f_2, label='MC first visit')
# plt.plot(x_cut, t_2, label='TD')

plt.plot(x, e_mean, label='MC every visit')
plt.plot(x, f_mean, label='MC first visit')
plt.plot(x, t1_mean, label='TD e=0.1')
plt.plot(x, t3_mean, label='TD e=0.3')
plt.plot(x, t5_mean, label='TD e=0.5')
plt.plot(x, t7_mean, label='TD e=0.7')


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

# plt.plot(x, t_median)
# plt.plot(x, e_median)
# plt.plot(x, f_median)

# plt.plot(x, t_median, color='#CC4F2B')
# plt.fill_between(x, t_25, t_75,
#                  alpha=0.3, edgecolor='#CC4F1B', facecolor='#FF9848')
# plt.fill_between(x, t_05, t_95,
#                  alpha=0.1, edgecolor='#CC4F1B', facecolor='#FF9848')


# plt.plot(x, e_median, color='#1B2ACC')
# plt.fill_between(x, e_25, e_75,
#                  alpha=0.3, edgecolor='#1B2ACC', facecolor='#089FFF')
# plt.fill_between(x, e_05, e_95,
#                  alpha=0.1, edgecolor='#1B2ACC', facecolor='#089FFF')


# plt.plot(x, f_median, color='#3F7F4C')
# plt.fill_between(x, f_25, f_75,
#                  alpha=0.3, edgecolor='#3F7F4C', facecolor='#7EFF99')
# plt.fill_between(x, f_05, f_95,
#                  alpha=0.1, edgecolor='#3F7F4C', facecolor='#7EFF99')
plt.show()

print('end')