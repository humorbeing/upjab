import matplotlib.pyplot as plt
import glob
import os

root = os.path.dirname(__file__)

def histogram_counter(target_folder):
    file_list = glob.glob(target_folder + '/**/*.png', recursive=True)
    counter_all = dict()
    counter_normal = dict()
    counter_abnormal = dict()
    for i in range(10):
        counter_all[i] = 0
        counter_normal[i] = 0
        counter_abnormal[i] = 0

    def group_num(file_name):
        return int(file_name.strip().split('/')[-2])

    checker = '/abnormal_'

    for file_name in file_list:
        num = group_num(file_name)
        counter_all[num] += 1
        if checker in file_name:
            counter_abnormal[num] += 1
        else:
            counter_normal[num] += 1
    
    return {
        'all': counter_all,
        'normal': counter_normal,
        'abnormal': counter_abnormal,
    }

target_folder = f'{root}/data/color/04_histogram/02_dataset_histogram'

counter_dataset = histogram_counter(target_folder)


d1 = tuple(range(10))
d_all = tuple(counter_dataset['all'].values())
d_normal = tuple(counter_dataset['normal'].values())
d_abnormal = tuple(counter_dataset['abnormal'].values())

p_abnormal = []

for i in d1:
    if d_all[i] != 0:
        p_abnormal.append(d_abnormal[i]/d_all[i]*100)
    else:
        p_abnormal.append(0)


fig=plt.figure(figsize=(6,5))
ax1 = fig.add_subplot(111)
t1 = ax1.bar(d1, d_all, color='black', label='Blind')

fig.legend(
    loc='upper left',
    bbox_to_anchor=(0.3, 0.88),
    # title='p_ :12345678912',
    )

# plt.xticks(myDF.index, myDF.Bin, rotation=60)

# plt.xlabel ('Group')
ax1.set_xlabel ('Histogram Group')
ax1.set_ylabel ('Number of Samples in a Group')


save_folder = os.path.dirname(__file__) + '/images'
os.makedirs(save_folder, exist_ok=True)
save_name = 'histogram_dataset_black'
# plt.savefig(save_folder + '/'+save_name+'.png')
plt.show()


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



t1 = ax1.bar(d1, d_all, color=abnomral_color, label='Abnormal')
t2 = ax1.bar(d1, d_normal, color=normal_color, label='Normal')

t3=ax2.scatter(d1, p_abnormal, marker='o',color=perc_piont_color, label='% of Abnormal')


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


# plt.show()  # This worked

save_folder = os.path.dirname(__file__) + '/images'
os.makedirs(save_folder, exist_ok=True)
save_name = 'histogram_dataset'
# plt.savefig(save_folder + '/'+save_name+'.png')
# [0.09, 0.82, 3.04, 10.51, 19.31, 32.62, 49.57, 69.23, 92.56, 99.52]

plt.show()

#######################

#######################

####################### gap dataset to labled

#######################

#######################

#######################



target_folder = f'{root}/data/color/04_histogram/labeled_histogram'

counter_labled = histogram_counter(target_folder)


d1 = tuple(range(10))
d_all = tuple(counter_labled['all'].values())
d_normal = tuple(counter_labled['normal'].values())
d_abnormal = tuple(counter_labled['abnormal'].values())

p_abnormal = []

for i in d1:
    if d_all[i] != 0:
        p_abnormal.append(d_abnormal[i]/d_all[i]*100)
    else:
        p_abnormal.append(0)

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



t1 = ax1.bar(d1, d_all, color=abnomral_color, label='Abnormal')
t2 = ax1.bar(d1, d_normal, color=normal_color, label='Normal')

t3=ax2.scatter(d1, p_abnormal, marker='o',color=perc_piont_color, label='% of Abnormal')


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


# plt.show()  # This worked

save_folder = os.path.dirname(__file__) + '/images'
os.makedirs(save_folder, exist_ok=True)
save_name = 'histogram_labeled'
# plt.savefig(save_folder + '/'+save_name+'.png')
# [0.0, 0.0, 0.0, 0, 0, 0, 0, 0, 0, 100.0]

plt.show()

#######################

#######################

####################### gap labeled to unlabled

#######################

#######################

#######################



target_folder = f'{root}/data/color/04_histogram/unlabeled_histogram'

counter_unlabled = histogram_counter(target_folder)


d1 = tuple(range(10))
d_all = tuple(counter_unlabled['all'].values())
d_normal = tuple(counter_unlabled['normal'].values())
d_abnormal = tuple(counter_unlabled['abnormal'].values())

p_abnormal = []

for i in d1:
    if d_all[i] != 0:
        p_abnormal.append(d_abnormal[i]/d_all[i]*100)
    else:
        p_abnormal.append(0)

fig=plt.figure(figsize=(6,5))
ax1 = fig.add_subplot(111)
t1 = ax1.bar(d1, d_all, color='black', label='Blind')

fig.legend(
    loc='upper left',
    bbox_to_anchor=(0.3, 0.88),
    # title='p_ :12345678912',
    )

# plt.xticks(myDF.index, myDF.Bin, rotation=60)

# plt.xlabel ('Group')
ax1.set_xlabel ('Histogram Group')
ax1.set_ylabel ('Number of Samples in a Group')


save_folder = os.path.dirname(__file__) + '/images'
os.makedirs(save_folder, exist_ok=True)
save_name = 'histogram_unlabeled_black'
# plt.savefig(save_folder + '/'+save_name+'.png')

plt.show()


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



t1 = ax1.bar(d1, d_all, color=abnomral_color, label='Abnormal')
t2 = ax1.bar(d1, d_normal, color=normal_color, label='Normal')

t3=ax2.scatter(d1, p_abnormal, marker='o',color=perc_piont_color, label='% of Abnormal')


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


# plt.show()  # This worked

save_folder = os.path.dirname(__file__) + '/images'
os.makedirs(save_folder, exist_ok=True)
save_name = 'histogram_unlabeled'
# plt.savefig(save_folder + '/'+save_name+'.png')

# [0.08927824121864798, 0.8214053350683148, 3.0363697030363697, 10.513447432762836, 19.305019305019304, 32.61943986820428, 49.57410562180579, 69.23076923076923, 92.5626515763945, 99.51417004048582]
plt.show()

print('done')