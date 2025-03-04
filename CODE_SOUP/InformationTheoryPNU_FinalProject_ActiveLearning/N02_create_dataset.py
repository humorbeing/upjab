
import os
import glob
import shutil


root = os.path.dirname(__file__)
working_folder = f'{root}/data/color'

target_folder = f'{working_folder}/01_pre_dataset/abnormal'
save_folder = f'{working_folder}/02_dataset/abnormal'
os.makedirs(save_folder, exist_ok=True)


file_list = glob.glob(target_folder + '/**/*.png', recursive=True)


counter = 0
prefix = 'abnormal'
for f_ in file_list:    
    # file_name = os.path.basename(f_)  # name of file
    # folder_path = os.path.dirname(f_)  # folder of file
    # save_folder_path = os.path.join(save_folder, folder_path)
    # os.makedirs(save_folder_path, exist_ok=True)
    shutil.copy(f_, f'{save_folder}/{prefix}_{counter:08d}.png')
    # shutil.copytree(source_dir, destination_dir)
    counter += 1



target_folder = f'{working_folder}/01_pre_dataset/normal'
save_folder = f'{working_folder}/02_dataset/normal'
os.makedirs(save_folder, exist_ok=True)


file_list = glob.glob(target_folder + '/**/*.png', recursive=True)


counter = 0
prefix = 'normal'
for f_ in file_list:    
    # file_name = os.path.basename(f_)  # name of file
    # folder_path = os.path.dirname(f_)  # folder of file
    # save_folder_path = os.path.join(save_folder, folder_path)
    # os.makedirs(save_folder_path, exist_ok=True)
    shutil.copy(f_, f'{save_folder}/{prefix}_{counter:08d}.png')
    # shutil.copytree(source_dir, destination_dir)
    counter += 1
print('done')