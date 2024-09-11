import glob
import shutil
from tqdm import tqdm
import os

def what_missing_in_folder(
    original_path,
    changed_path,
    file_type_list=['JPG','jpg']
    ):
    original_list = []
    for ty_ in file_type_list:
        temp_file_list = glob.glob(original_path + f'/**/*.{ty_}', recursive=True)    
        original_list.extend(temp_file_list)
    
    

    changed_list = []
    for ty_ in file_type_list:
        temp_file_list = glob.glob(changed_path + f'/**/*.{ty_}', recursive=True)    
        changed_list.extend(temp_file_list)

    
    def get_name(_f):
        return _f.split('/')[-1]

    changed_names = []
    for _f in changed_list:
        changed_names.append(get_name(_f))

    change_name = changed_path.split('/')[-1]

    save_folder = f'{original_path}_OOOUTTTOFFF_{change_name}'
    os.makedirs(save_folder, exist_ok=True)
    counter = 0
    for _f in tqdm(original_list):
        if get_name(_f) in changed_names:
            counter += 1
        else:        
            shutil.copy(_f, save_folder)


if __name__ == '__main__':
    original_path = '../../example_data/videos/fishes/not_crowd'
    changed_path = '../../example_data/videos/fishes/not_crowd_0.80'

    what_missing_in_folder(
        original_path=original_path,
        changed_path=changed_path,
        file_type_list=['mp4']
    )


    print('')
