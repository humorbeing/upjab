import matplotlib.pyplot as plt
import glob
import os

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


#######################

#######################

####################### gap labeled to unlabled

#######################

#######################

#######################
# p_unlabeled = 10.44

root = os.path.dirname(__file__)

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




def count_one(one_folder):
    file_list = glob.glob(one_folder + '/**/*.png', recursive=True)

    num_all = len(file_list)
    num_abnormal = 0
    num_normal = 0

    checker = '/abnormal_'

    for file_name in file_list:
        if checker in file_name:
            num_abnormal += 1
        else:
            num_normal += 1
    
    prob = num_abnormal / num_all * 100
    return {'all': num_all, 'normal': num_normal, 'abnormal': num_abnormal, 'prob': prob}

one_folder = f'{root}/data/color/03_randomsample/unlabeled'
first_count = count_one(one_folder)


d_all = []
d_normal = []
p_sample = []


for i in range(10):
    one_count = count_one(f'{root}/data/color/05_group_sample/{i}_labeled')
    d_all.append(one_count['all'])
    d_normal.append(one_count['normal'])
    p_sample.append(one_count['prob'])


one_ = f'{root}/data/color/05_group_sample/unlabeled_histogram_labeled'
last_one = count_one(one_)

fig=plt.figure(figsize=(6,5))
ax1 = fig.add_subplot(111)
ax1.set_ylim(-3,d_all[0]*1.5)
ax2 = ax1.twinx()
ax2.set_ylim(-3,103)

# Colors
normal_color = '#d2d6dc'
abnomral_color = '#446d92'
perc_piont_color = '#b0353d'
line_color = '#ff1c5d'
# font_color = 'black'



t1 = ax1.bar(d1, d_all, color=abnomral_color, label='Abnormal')
t2 = ax1.bar(d1, d_normal, color=normal_color, label='Normal')

t1 = ax1.bar(8.5, last_one['all'], width=0.5, color='blue', label='Ran Ab')
t2 = ax1.bar(8.5, last_one['normal'], width=0.5, color='green', label='Ran Nor')

t3=ax2.scatter(d1, p_abnormal, marker='o',color=perc_piont_color, label='True Group')

ax2.scatter(8.5, first_count['prob'], marker='*',color=perc_piont_color, label='True Dataset')

# p_sample = [
#     1,5,17,31,47,73,84,95,97,99
# ]
t3=ax2.scatter(d1, p_sample, marker='o',color='black', label='Sample Group')

ax2.scatter(8.5, last_one['prob'], marker='*',color='black', label='Dataset Group')

fig.legend(
    # loc='upper left',
    # bbox_to_anchor=(0.3, 0.88),
    # title='p_ :12345678912',
    ncol=4
    )

# plt.xticks(myDF.index, myDF.Bin, rotation=60)

# plt.xlabel ('Group')
# ax1.set_xlabel ('Histogram Group')
# ax1.set_ylabel ('Number of Samples in a Group')
ax2.set_ylabel ('% of Abnormal Sample in a Group')
ax2.grid(True)
# plt.title ('Subject mark of student')
# ax1.set_yscale('log')
# ax1.set_yscale('linear')


# plt.show()  # This worked

save_folder = os.path.dirname(__file__) + '/images'
os.makedirs(save_folder, exist_ok=True)
save_name = 'unlabeled_sample'
plt.savefig(save_folder + '/'+save_name+'.png')

# [0.08927824121864798, 0.8214053350683148, 3.0363697030363697, 10.513447432762836, 19.305019305019304, 32.61943986820428, 49.57410562180579, 69.23076923076923, 92.5626515763945, 99.51417004048582]


print('done')