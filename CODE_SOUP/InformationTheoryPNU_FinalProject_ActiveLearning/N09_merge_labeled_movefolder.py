import glob
import os
import shutil



def merge_labeled(target_folder, first_labeled):
    
    file_list1 = glob.glob(first_labeled + '/**/*.png', recursive=True)
    file_list2 = glob.glob(target_folder + '/**/*.png', recursive=True)
    file_list1.extend(file_list2)

    checker = 'abnormal_'
    for f_ in file_list1:
        if checker in f_:
            save_folder = f'{target_folder}_merged/abnormal'
            os.makedirs(save_folder, exist_ok=True)
            shutil.copy(f_, save_folder)
        else:
            save_folder = f'{target_folder}_merged/normal'
            os.makedirs(save_folder, exist_ok=True)
            shutil.copy(f_, save_folder)

root = os.path.dirname(__file__)
first_labeled = f'{root}/data/color/03_randomsample/labeled'
target_folder = f'{root}/data/color/05_group_sample/0_labeled'

root1 = f'{root}/data/color/05_group_sample'
# merge_labeled(target_folder, first_labeled)


for i in os.scandir(root1):
    folder_name = i.name
    folder_path = os.path.join(root1, folder_name)
    
    merge_labeled(folder_path, first_labeled)


# Then
# create folder 06_labeled
# move 8_labeled_merged to 06_labeled
# move unlabeled_histogram_labeled_merged to 06_labeled



print('done')