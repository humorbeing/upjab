try:
    from .get_file_path_list import get_file_path_list   
    # print("Using relative import")
except ImportError:
    from get_file_path_list import get_file_path_list
    # print("Using absoloute import")

import shutil
import os



def collect_to_one_folder(
    target_folder,
    file_extends=['jpg', 'JPG'],
):
    
    file_list = get_file_path_list(target_folder, file_extends=file_extends)   

    
    save_folder_path = target_folder + '_onefolder'
    os.makedirs(save_folder_path, exist_ok=True)
    
    for f_ in file_list:
        
        
        shutil.copy(f_, save_folder_path)  # dst can be folder

if __name__ == '__main__':
    
    target_folder = 'example_data/text/white_histogram'
    collect_to_one_folder(
        target_folder=target_folder,
        file_extends=['jpg', 'txt'],        
    )      
    print('End')