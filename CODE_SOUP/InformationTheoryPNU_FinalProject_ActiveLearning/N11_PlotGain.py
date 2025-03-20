


import glob
import matplotlib.pyplot as plt
import numpy as np
import os



def get_aucroc(target_folder):
    file_list = glob.glob(target_folder + '/**/*.pkl', recursive=True)
    file_list.sort()

    def g_aucroc(file_list):
        return float(file_list.split('/')[-1].split('_')[-2])

    aucroc_list = []
    for file_ in file_list:
        aucroc_list.append(g_aucroc(file_))
    
    return aucroc_list

root = os.path.dirname(__file__)


m4_dog_aucroc = get_aucroc(f'{root}/model_weights')


normal_color = '#d2d6dc'
abnomral_color = '#446d92'
perc_piont_color = '#b0353d'
line_color = '#ff1c5d'


def plot_aucroc_gain(auc_list, save_name='test'):
    base_auc = auc_list[10]
    auc_points = auc_list[:10]
    random_auc = auc_list[11]
    auc_points = np.array(auc_points)
    gain_list = (auc_points -base_auc)/base_auc * 100
    random_gain = (random_auc - base_auc)/base_auc * 100

    fig=plt.figure(figsize=(6,5))
    ax1 = fig.add_subplot(111)

    d1 = list(range(10))
    t1 = ax1.bar(d1, gain_list, color=abnomral_color, label='Abnormal')
    t1 = ax1.bar(8.5, random_gain, width=0.5, color='blue', label='Ran Ab')

    plt.title(f'{save_name} AUCROC Gain')
    save_folder = os.path.dirname(__file__) + '/images'
    os.makedirs(save_folder, exist_ok=True)
    
    # plt.savefig(f'{save_folder}/{save_name}_gain.png')
    plt.show()


def plot_aucroc_prob(auc_list, save_name='test'):
    base_auc = auc_list[10]
    auc_points = auc_list[:10]
    random_auc = auc_list[11]
    auc_points = np.array(auc_points)
    # gain_list = (auc_points -base_auc)/base_auc * 100
    # random_gain = (random_auc - base_auc)/base_auc * 100

    fig=plt.figure(figsize=(6,5))
    ax1 = fig.add_subplot(111)
    ax1.set_ylim(base_auc-0.2,1.01)
    ax1.axhline(y=base_auc, color='black', label=f'Model1 AUCROC: {base_auc:.03f}')
    d1 = list(range(10))

    t1 = ax1.bar(d1, auc_points, color=perc_piont_color)
    t1 = ax1.bar(8.5, random_auc, width=0.5, color=line_color)
    # t1 = ax1.bar(d1, [base_auc]*10, color='black', label='Abnormal')
    fig.legend()

    plt.title(f'{save_name} AUCROC')
    save_folder = os.path.dirname(__file__) + '/images'
    os.makedirs(save_folder, exist_ok=True)
    
    # plt.savefig(f'{save_folder}/{save_name}_aucroc.png')
    plt.show()

# plot_aucroc_gain(m1_7_aucroc, 'm1_7')
# plot_aucroc_prob(m1_7_aucroc, 'm1_7')

# plot_aucroc_gain(m2_5_aucroc, 'm2_5')
# plot_aucroc_prob(m2_5_aucroc, 'm2_5')

# plot_aucroc_gain(m3_dress_aucroc, 'm3_dress')
# plot_aucroc_prob(m3_dress_aucroc, 'm3_dress')

plot_aucroc_gain(m4_dog_aucroc, 'm4_dog')
plot_aucroc_prob(m4_dog_aucroc, 'm4_dog')
print('done')