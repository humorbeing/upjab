






import glob
import random
random.seed(42)  # for reproducibility


import shutil
import os




def random_sample(target_folder, CUTTING = 100):

    file_list = glob.glob(target_folder + '/**/*.png', recursive=True)

    
    random.shuffle(file_list)

    

    labeled = file_list[:CUTTING]
    

    

    checker = 'abnormal_'
    for f_ in labeled:
        if checker in f_:
            save_folder = f'{target_folder}_labeled/abnormal'
            os.makedirs(save_folder, exist_ok=True)
            shutil.copy(f_, save_folder)
        else:
            save_folder = f'{target_folder}_labeled/normal'
            os.makedirs(save_folder, exist_ok=True)
            shutil.copy(f_, save_folder)


root = os.path.dirname(__file__)
target_folder = f'{root}/data/color/04_histogram/unlabeled_histogram'


random_sample(target_folder)
for i in range(10):
    tf = f'{target_folder}/{i}'
    random_sample(tf)




# Then
# create folder 05_group_sample

print('done')