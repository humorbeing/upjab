
from upjab import hi

hi()

from upjab.tool import shuffle_split_folder
from upjab.tool import what_missing_in_folder

target_folder = '../example_data/videos/fishes/crowd'
shuffle_split_folder(
    target_folder=target_folder,
    file_type_list=['mp4'],
    random_seed=5,
    split_ratio=0.60
)
original_path = '../example_data/videos/fishes/crowd'
changed_path = '../example_data/videos/fishes/crowd_0.60'

what_missing_in_folder(
    original_path=original_path,
    changed_path=changed_path,
    file_type_list=['mp4']
)
print('end')