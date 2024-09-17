


from tool.experiment_configurations.v0001.args_setup import args_setup
from tool.args_setup_and_logger import args_setup_and_logger
args = args_setup("From_Experiment_Args_Setup")
logging = args_setup_and_logger(args)
logging.info(f'fake accuracy: 0.58 fake auc-roc: 0.88 fake loss: 0.42')
logging.info("done")
with open(args.checkpoint_save_path, 'w') as f:
    f.write('fake checkpoint')
print('')


import torch
from vision.i3d_video_feature import I3D_Video_Feature

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
i3d_video_feature = I3D_Video_Feature(
    ckpt_path=None,        
    device=device,
    local_weight=False)

video_path = 'example_data/videos/fishes/crowd/00000001.mp4'
video_feature = i3d_video_feature.extract_video(
    video_path=video_path,
    is_ten_crop=True,
    target_number_segment=12,        
    return_numpy=True)

print(video_feature.shape)

video_feature = i3d_video_feature.extract_video(
    video_path=video_path,
    is_ten_crop=False,
    target_number_segment=7,        
    return_numpy=False)

print(video_feature.shape)

video_path = 'example_data/videos/fishes/crowd/00000103_notcrowd.mp4'
video_feature = i3d_video_feature.extract_video(
    video_path=video_path,
    is_ten_crop=True,
    target_number_segment=4,        
    return_numpy=False)

print(video_feature.shape)


folder_path = 'example_data/videos/fishes/not_crowd'
i3d_video_feature.extract_folder(folder_path=folder_path)

from vision.read_video import read_video


video_path = 'example_data/videos/fishes/crowd/00000001.mp4'
vid = read_video(video_path=video_path)
vid = read_video(video_path=video_path, echo=True)
# vid = np.random.randint(0, 256, size=(105,360,480,3))


video_path = 'example_data/videos/fishes/crowd/00000103_notcrowd.mp4'
vid = read_video(
    video_path=video_path,
    is_cv2=True,
    feeding_fps=None,
    echo=True)

vid = read_video(
    video_path=video_path,
    is_cv2=False,
    feeding_fps=None,
    echo=True)

vid = read_video(
    video_path=video_path,
    is_cv2=True,
    feeding_fps=15,
    echo=True)





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

from tool.setup_logger import setup_logger

filepath = 'example_data/example.log'
logger = setup_logger(filepath)
logger.info('hi from upjab logger')
