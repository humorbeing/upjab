import shutil
import os



def remove_folder(target_folder):
    if os.path.isdir(target_folder):
        shutil.rmtree(target_folder)
    else:
        print(f'folder [{target_folder}] does not exist.')