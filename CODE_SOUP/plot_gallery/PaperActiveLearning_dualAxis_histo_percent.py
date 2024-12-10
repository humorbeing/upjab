
log_path = 'CODE_SOUP/plot_gallery/data/pre-histogram_final_pick_low_all_train.txt'

show_figure = True
save_figure = False
save_name = '1'
save_folder = 'show_figures/figures'
# import os
# os.makedirs(save_folder, exist_ok=True)
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



norm_count = []
abnorm_count = []
all_count = []
percent_count = []
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
    percent_count.append(pers*100)


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.lines as mlines

# sns.set_theme(style="darkgrid")

myDF = pd.DataFrame()
# ax.set_title('Need a name')

import numpy as np

d1 = tuple(range(10))
d2 = np.array(all_count)
d2 = all_count
d3 = np.array(norm_count)
myDF['abnormal_count'] = abnorm_count
d4 = percent_count

fig=plt.figure(figsize=(6,5))
ax1 = fig.add_subplot(111)
ax2 = ax1.twinx()
ax2.set_ylim(0,100)

# Colors
normal_color = '#d2d6dc'
abnomral_color = '#446d92'
perc_piont_color = '#b0353d'
line_color = '#ff1c5d'
# font_color = 'black'


# sns.barplot(x=myDF.index,y='all_count',data=myDF,color='blue',ax=ax1, estimator=sum, ci=None)
t1 = ax1.bar(d1, d2, color=abnomral_color, label='Abnormal')
t2 = ax1.bar(d1, d3, color=normal_color, label='Normal')

# t1 = ax1.bar(x=myDF.index,y='all_count',data=myDF,color=abnomral_color,ax=ax1)
# t2 = sns.barplot(x=myDF.index,y='normal_count',data=myDF,color=normal_color,ax=ax1)
# sns.lineplot(x=myDF.index,y='Abnormals',data=myDF,marker='s',color='orange',ax=ax2)


t3=ax2.scatter(d1, d4, marker='o',color=perc_piont_color, label='% of Abnormal')

s_points_x = [0, 8, 9]
s_points_y = [percent_count[0], percent_count[8], percent_count[9]]

t3=ax2.scatter(s_points_x, s_points_y, s=50, marker='+',color='black', label='Key Points')
# ax2.scatter(0, percent_count[0], s=200, alpha=0.5, marker='o', color=line_color)
# ax2.scatter(8, percent_count[8], s=200, alpha=0.5, marker='o', color=line_color)
# ax2.scatter(9, percent_count[9], s=200, alpha=0.5, marker='o', color=line_color)
xmin, xmax = [0, 8]
ymin, ymax = [percent_count[0], percent_count[8]]
l = mlines.Line2D([xmin,xmax], [ymin,ymax], color=line_color, linewidth=5, alpha=0.4, label='Linear Trend')
ax2.add_line(l)
x_move = 0.1
y_move = 2

ax2.text(0+x_move, percent_count[0]+y_move, s='p$_0$')
ax2.text(8+x_move, percent_count[8]+y_move, s='p$_8$')
ax2.text(9+x_move, percent_count[9]+y_move, s='p$_9$')



import matplotlib.patches as mpatches
top_bar = mpatches.Patch(color=abnomral_color, label='Abnormal')
bottom_bar = mpatches.Patch(color=normal_color, label='Normal')
# plt.legend(handles=[top_bar, bottom_bar, t1, t2,t3])
# plt.legend(handles=[t1,t2])
fig.legend(
    loc='upper left',
    bbox_to_anchor=(0.3, 0.88),
    # title='p_ :12345678912',
    )

# plt.xticks(myDF.index, myDF.Bin, rotation=60)

# plt.xlabel ('Group')
ax1.set_xlabel ('Histogram Group')
ax1.set_ylabel ('Number of Samples in a Group')
ax2.set_ylabel ('% of Abnormal Sample in a Group')
ax2.grid(True)
# plt.title ('Subject mark of student')
# ax1.set_yscale('log')
# ax1.set_yscale('linear')

if show_figure:
    plt.show()  # This worked


# # NO! doesn't work
# plt.savefig(f'{save_folder}/{save_name}.pdf')
# plt.savefig(f'{save_folder}/{save_name}.png')
# plt.savefig(f'{save_folder}/{save_name}.svg')


if save_figure:
    fig.savefig(f'{save_folder}/{save_name}.pdf')
    fig.savefig(f'{save_folder}/{save_name}.png')
    fig.savefig(f'{save_folder}/{save_name}.svg')

# fig.show()  # NO! doesn't work

print()




save_name = '1'
save_folder = 'show_figures/figures'
# import os
# os.makedirs(save_folder, exist_ok=True)
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



norm_count = []
abnorm_count = []
all_count = []
percent_count = []
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
    percent_count.append(pers*100)


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.lines as mlines

# sns.set_theme(style="darkgrid")

myDF = pd.DataFrame()
# ax.set_title('Need a name')

myDF['Bin'] = list(range(10))
myDF['all_count'] = all_count
myDF['normal_count'] = norm_count
myDF['abnormal_count'] = abnorm_count
myDF['percentage'] = percent_count

fig=plt.figure(figsize=(6,5))
ax1 = fig.add_subplot(111)
ax2 = ax1.twinx()
ax2.set_ylim(0,100)

# Colors
normal_color = 'lightblue'
abnomral_color = 'red'
perc_piont_color = 'red'
line_color = 'red'
# font_color = 'black'


# sns.barplot(x=myDF.index,y='all_count',data=myDF,color='blue',ax=ax1, estimator=sum, ci=None)
# t1 = sns.barplot(x=myDF.index,y='all_count',data=myDF,color=abnomral_color,ax=ax1, label='All')
# t2 = sns.barplot(x=myDF.index,y='normal_count',data=myDF,color=normal_color,ax=ax1, label='Normal')

t1 = sns.barplot(x=myDF.index,y='all_count',data=myDF,color=abnomral_color,ax=ax1)
t2 = sns.barplot(x=myDF.index,y='normal_count',data=myDF,color=normal_color,ax=ax1)
# sns.lineplot(x=myDF.index,y='Abnormals',data=myDF,marker='s',color='orange',ax=ax2)

xmin, xmax = [0, 8]
ymin, ymax = [percent_count[0], percent_count[8]]
l = mlines.Line2D([xmin,xmax], [ymin,ymax], color=line_color, linewidth=5, alpha=0.5)
ax2.add_line(l)
t3=sns.scatterplot(x=myDF.index,y='percentage',data=myDF,marker='o',color=perc_piont_color,ax=ax2)

# ax2.scatter(0, percent_count[0], s=200, alpha=0.5, marker='o', color=line_color)
# ax2.scatter(8, percent_count[8], s=200, alpha=0.5, marker='o', color=line_color)
# ax2.scatter(9, percent_count[9], s=200, alpha=0.5, marker='o', color=line_color)

x_move = 0.1
y_move = 2

ax2.text(0+x_move, percent_count[0]+y_move, s='p$_0$')
ax2.text(8+x_move, percent_count[8]+y_move, s='p$_8$')
ax2.text(9+x_move, percent_count[9]+y_move, s='p$_9$')

tt1, = plt.plot(0,0, alpha=0.9, marker='o',color=perc_piont_color)

import matplotlib.patches as mpatches
top_bar = mpatches.Patch(color=abnomral_color, label='Abnormal')
bottom_bar = mpatches.Patch(color=normal_color, label='Normal')
plt.legend(handles=[top_bar, bottom_bar, t1, t2,t3])
# plt.legend(handles=[tt1])
# plt.legend()

plt.xticks(myDF.index, myDF.Bin, rotation=60)

# plt.xlabel ('Group')
ax1.set_xlabel ('Histogram Group')
ax1.set_ylabel ('Counts')
ax2.set_ylabel ('% of Abnormal Sample')
# plt.title ('Subject mark of student')
# ax1.set_yscale('log')
# ax1.set_yscale('linear')


plt.show()  # This worked


# # NO! doesn't work
# plt.savefig(f'{save_folder}/{save_name}.pdf')
# plt.savefig(f'{save_folder}/{save_name}.png')
# plt.savefig(f'{save_folder}/{save_name}.svg')



# fig.savefig(f'{save_folder}/{save_name}.pdf')
# fig.savefig(f'{save_folder}/{save_name}.png')
# fig.savefig(f'{save_folder}/{save_name}.svg')

# fig.show()  # NO! doesn't work

print()
