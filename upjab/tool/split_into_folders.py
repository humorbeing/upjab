# from tqdm import tqdm
import os
try:
    from .get_file_path_list import get_file_path_list   
    # print("Using relative import")
except ImportError:
    from get_file_path_list import get_file_path_list
    # print("Using absoloute import")
# from get_file_path import get_file_path
import shutil




def split_into_folders(
    target_folder,
    file_extends=['jpg', 'JPG'],
    cutting=50,
    is_shuffle=True
    ):

    file_list = get_file_path_list(target_folder, file_extends=file_extends)
    
    if is_shuffle:
        import random
        random.seed(5)
        random.shuffle(file_list)   

    steps = int(len(file_list)/cutting)
    save_folder = target_folder + '_split'
    for i in range(steps):
        save_path = f'{save_folder}/{i:04d}'
        os.makedirs(save_path, exist_ok=True)
        selected = file_list[cutting*i: cutting*(i+1)]
        for f_ in selected:
            shutil.copy(f_, save_path)
    
    if len(file_list) % cutting == 0:
        pass
    else:
        i = steps
        save_path = f'{save_folder}/{i:04d}'
        os.makedirs(save_path, exist_ok=True)
        selected = file_list[cutting*i:]
        for f_ in selected:
            shutil.copy(f_, save_path)
    # print('end')


if __name__ == '__main__':
    target_folder = '../../example_data/images/disease_image/images_big_image_dataset/disease'
    split_into_folders(
        target_folder=target_folder,
        file_extends=['jpg', 'JPG'],
        cutting=3,
        is_shuffle=True
    )      
    print('End')