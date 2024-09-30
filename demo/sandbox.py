path = 'example_data/text/long-tail_dataset/white'

import glob
target_folder = path
file_list = glob.glob(target_folder + '/**/*.*', recursive=True)
# file_list2 = glob.glob(target_folder + '/**/*.jpg', recursive=True)
# file_list.extend(file_list2)

import shutil
import os
save_folder = './python_files'
counter = 4
for f_ in file_list:
    counter += 1
    orginal_folder = f_.split('/')[:-1]
    file_name = os.path.basename(f_)  # name of file
    folder_path = os.path.dirname(f_)  # folder of file
    save_folder_path = os.path.join(save_folder, folder_path)
    os.makedirs(save_folder_path, exist_ok=True)
    shutil.copy(f_, f'{save_folder_path}/{counter:03d}.txt')  # dst can be folder
    # shutil.copytree(source_dir, destination_dir)
