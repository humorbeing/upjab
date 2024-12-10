
import matplotlib.pyplot as plt
import numpy as np




def draw_line(log_path, ax, color):
    with open(log_path, 'r') as f:
        lines = f.readlines()


    histogram = {}

    for i in range(10):
        histogram[f'{i}'] = []

    def get_num(input_string):
        div = input_string.strip().split(' ')
        file_name = div[0]
        label = int(float(div[2]))
        pred = float(div[-1])
        keyss = div[-1][2]
        return file_name, label, pred, keyss

    for line_ in lines:
        file_name, label, pred, keyss = get_num(line_)
        histogram[keyss].append([file_name, label, pred])






    species = [str(i) for i in range(10)]

    norm_count = []
    abnorm_count = []
    all_count = []
    percent_count = []
    perc = []

    sex_counts = {}
    for i in range(10):
        k_ = str(i)
        n_count = 0
        a_count = 0
        for ele in histogram[k_]:
            if ele[1] == 0:
                n_count += 1
            else:
                a_count += 1
        
        norm_count.append(n_count)
        abnorm_count.append(a_count)
        all_count.append(n_count+a_count)
        if n_count+a_count == 0:
            pers = 0.0
        else:
            pers = a_count/(n_count+a_count)
        percent_count.append(f'{pers:.3f}')
        perc.append(pers*100)
        # print()

    x_s = list(range(10))
    x1, y1 = x_s[0], perc[0]
    x2, y2 = x_s[8], perc[8]

    ax.scatter(x_s, perc, color=color, marker='+', label='Ground Truth')
    # plt.plot([x1,x2],[y1,y2])



color_gt = 'black'
color_sample = 'blue'
color_com = 'red'

log_path = 'uu/plot_gallery/data/pre-histogram_final_pick_low_all_train.txt'


fig=plt.figure(figsize=(6,5))
ax1 = fig.add_subplot(111)
ax1.set_ylim(0,100)
draw_line(log_path, ax1, color=color_gt)
ax1.scatter((0),(4),c='black',
            marker='p', label='Key Point (Group 0)')

ax1.scatter((8,9),(44,84),c=color_sample,
            marker='>', label='Key Point (Initial Samples)')
ax1.plot((0,8),(4,44),'-',color=color_sample, linewidth=1, alpha=0.3)
# plt.plot((8,9),(0.44,0.84),'r-')


# plt.plot((8,9),(0.567,0.9316),'r-')

xs = (0, 8)
ys = (4, 44)
x_ = list(range(1,8))
y_ = []
for i in x_:
    y_.append(np.interp(i, xs,ys))
ax1.scatter(x_,y_,c=color_sample, marker='4', label='Linear Trend (Initial Samples)')


ax1.scatter((8,9),(56.7,93.16),c=color_com,
            marker='*', label='Key Point (Compensation)')
ax1.plot((0,8),(4,56.7),'-',color=color_com, linewidth=2, alpha=0.3)
print(y_)
xs = (0, 8)
ys = (4, 56.7)
x_ = list(range(1,8))
y_ = []
for i in x_:
    y_.append(np.interp(i, xs,ys))
ax1.scatter(x_,y_,c=color_com, marker='x', label='Linear Trend (Compensation)')


print(y_)
d1 = tuple(range(10))
plt.xticks(d1, d1)
plt.legend()
ax1.grid(axis='y')
ax1.set_xlabel('Histogram Group')
ax1.set_ylabel('% of Abnormal Sample in a Histogram Group')

save_folder = 'compensation'
save_name = 'compensate'
# fig.savefig(f'{save_folder}/{save_name}.pdf')
# fig.savefig(f'{save_folder}/{save_name}.png')
# fig.savefig(f'{save_folder}/{save_name}.svg')

plt.show()

# plt.savefig('realshow06_compensate.png')
# plt.savefig('realshow06_compensate.pdf')
print('')