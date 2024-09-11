from tool.histogram_folder import histogram_folder


target_folder = 'example_data/text/white'
import random
random.seed(5)

def fake_detect(video_path):
    return random.random()

histogram_folder(
    target_folder=target_folder,
    detect=fake_detect,
    file_type_list=['txt']
)

from tool.collect_to_one_folder import collect_to_one_folder

target_folder = 'example_data/text/white_histogram'
collect_to_one_folder(
    target_folder=target_folder,
    file_extends=['jpg', 'txt'],        
)


from tool.get_file_path_list import get_file_path_list
from tool.file_path_get_last_folder import file_path_get_last_folder


target_folder = 'example_data/images'
file_list = get_file_path_list(target_folder, file_extends=['jpg', 'JPG', 'json'])
for f_ in file_list:
    new_file_path = file_path_get_last_folder(f_, '_ALLGOOD')
    print(f_)
    print(new_file_path)
    print('--')


from tool.only_python_files import only_python_files

target_folder='example_data/images'
file_extends=['py']

only_python_files(
    target_folder=target_folder,
    file_extends=file_extends,
    
)

from tool.shuffle_split_folder import shuffle_split_folder

target_folder = 'example_data/text/white'
shuffle_split_folder(
    target_folder=target_folder,
    file_type_list=['txt'],
    random_seed=1,
    split_ratio=0.2
)

from tool.split_into_folders import split_into_folders
target_folder = 'example_data/text/black'
split_into_folders(
    target_folder=target_folder,
    file_extends=['jpg', 'txt'],
    cutting=3,
    is_shuffle=True,
    random_seed=5
)      


from tool.what_missing_in_folder import what_missing_in_folder
original_path = 'example_data/text/black'
changed_path = 'example_data/text/black_split/0000'

what_missing_in_folder(
    original_path=original_path,
    changed_path=changed_path,
    file_type_list=['txt']
)


import subprocess

subprocess.run(["ls", "-l"])
# subprocess.run(["ls -l"])  # error
subprocess.run(["ls", "demo/", "-l"])

subprocess.run(["chmod", "+x", "example_data/images/disease_image/run.sh"])
subprocess.run(["./run.sh"], shell=True, cwd="example_data/images/disease_image")
# subprocess.run(["cd", "example_data/images/disease_image/", "&&", "./run.sh"], shell=True)  # not working
# subprocess.run(["./run.sh"])  # error