
import shutil
import os
from upjab import get_file_list    


def only_python_files(    
    target_folder,
    file_extends=['py', 'ipynb'],    
    ):

    """
    Copies all files with specified extensions from the target folder to a new folder.
    This function scans the specified target folder for files with the given extensions,
    creates a new folder named `<target_folder>_OnlyPythonFile`, and copies the matching
    files into the corresponding subdirectories within the new folder.
    Args:
        target_folder (str): The path to the folder where the search for files will begin.
        file_extends (list, optional): A list of file extensions to filter by. Defaults to ['py', 'ipynb'].
    Instructions:
        1. Ensure the `target_folder` exists and contains files to process.
        2. The `file_extends` parameter should include valid file extensions without the leading dot.
        3. The function will create a new folder named `<target_folder>_OnlyPythonFile` in the same directory as `target_folder`.
        4. Subdirectory structure within `target_folder` will be preserved in the new folder.
        5. The function uses `shutil.copy` to copy files, so ensure sufficient disk space is available.
    Returns:
        None
    Example:
        only_python_files('/home/user/projects', file_extends=['py', 'ipynb'])
        # This will copy all `.py` and `.ipynb` files from `/home/user/projects` to
        # `/home/user/projects_OnlyPythonFile`, preserving the folder structure.
    """

    file_list = get_file_list(target_folder=target_folder,file_extends=file_extends)

    print(f'Found {len(file_list)} python files in {target_folder}.')
    save_folder = target_folder + '_OnlyPythonFile'
    front_cut = len(target_folder.split('/'))
    for f_ in file_list:
        orginal_folder = f_.split('/')[front_cut:-1]
        save_folder_path = os.path.join(save_folder, *orginal_folder)
        os.makedirs(save_folder_path, exist_ok=True)
        shutil.copy(f_, save_folder_path)  # dst can be folder

    print(f'Copied {len(file_list)} python files to {save_folder}.')

if __name__ == '__main__':
    target_folder='extra_packages'
    # file_extends=['py', 'ipynb']
    
    only_python_files(
        target_folder=target_folder,
        # file_extends=file_extends,        
    )
    # print('end')
