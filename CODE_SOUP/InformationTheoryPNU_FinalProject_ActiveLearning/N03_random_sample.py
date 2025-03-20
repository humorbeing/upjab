
import os
import glob
import shutil
import os
import random


root = os.path.dirname(__file__)
working_folder = f'{root}/data/color'

target_folder = f'{working_folder}/02_dataset'
save_folder = f'{working_folder}/03_randomsample'



file_list = glob.glob(target_folder + '/**/*.png', recursive=True)


random.seed(42)  # for reproducibility
random.shuffle(file_list)

CUTTING = 500

labeled = file_list[:CUTTING]
unlabeled = file_list[CUTTING:]



checker = 'abnormal_'
for f_ in labeled:
    if checker in f_:
        sf = f'{save_folder}/labeled/abnormal'
        os.makedirs(sf, exist_ok=True)
        shutil.copy(f_, sf)
    else:
        sf = f'{save_folder}/labeled/normal'
        os.makedirs(sf, exist_ok=True)
        shutil.copy(f_, sf)


for f_ in unlabeled:
    if checker in f_:    
        sf = f'{save_folder}/unlabeled/abnormal'    
        os.makedirs(sf, exist_ok=True)
        shutil.copy(f_, sf)
    else:
        sf = f'{save_folder}/unlabeled/normal'
        os.makedirs(sf, exist_ok=True)
        shutil.copy(f_, sf)




print('done')