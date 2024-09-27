
import shutil
import os

try:    
    from .get_file_path_list import get_file_path_list    
    # print("Using relative import")
except ImportError:
    from get_file_path_list import get_file_path_list    
    # print("Using absoloute import")


def only_python_files(
    target_folder='.',
    file_extends=['py'],
    
    ):
    '''
        
    target_folder='../../example_data/images'
    file_extends=['py']
    
    only_python_files(
        target_folder=target_folder,
        file_extends=file_extends,
        
    )
            
    '''
    file_list = get_file_path_list(target_folder=target_folder,file_extends=file_extends)

    print(f'Found {len(file_list)} files.')
    save_folder = target_folder + '_OnlyPythonFile'
    front_cut = len(target_folder.split('/'))
    for f_ in file_list:
        orginal_folder = f_.split('/')[front_cut:-1]
        save_folder_path = os.path.join(save_folder, *orginal_folder)
        os.makedirs(save_folder_path, exist_ok=True)
        shutil.copy(f_, save_folder_path)  # dst can be folder


if __name__ == '__main__':
    target_folder='example_data/images'
    file_extends=['py']
    
    only_python_files(
        target_folder=target_folder,
        file_extends=file_extends,
        
    )
    print('end')
