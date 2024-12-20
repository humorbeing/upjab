
from upjab import hi

hi()


from upjab.vision.video_to_image import video_to_image
video_path = 'example_data/videos/fishes/crowd/00000001.mp4'
video_to_image(video_path)

from upjab.vision.VideoClips import VideoClips

video_path = 'example_data/videos/fishes/crowd/00000103_notcrowd.mp4'
vc = VideoClips(
    video_path,
    frame_rate=15
)
temp11 = vc.get_clip(0)
print(f'{len(vc)=}')


from upjab.tool.timer import timethis
import time
with timethis as tt:
    time.sleep(2)

print(f'Print Elapsed time: {timethis.interval:>20.2f} seconds')
print(f'Print Elapsed time: {tt.interval:>20.2f} seconds')

with timethis:
    time.sleep(1.5)
print(f'Print Elapsed time: {timethis.interval:>20.2f} seconds')

from upjab.datasets.get_class_distribution import get_class_distribution
import torchvision

dataset_root = 'example_data/text/long-tail_dataset'

def whatis(x):
    return 0
dataset = torchvision.datasets.DatasetFolder(root=dataset_root, loader=whatis, extensions=['.txt'], transform=None)

print("Distribution of classes: \n", get_class_distribution(dataset))

print(dataset.classes)
print(dataset.class_to_idx)

print('done')

from torchvision.transforms import v2
from upjab.vision.torchvision_plot import torchvision_transform_plot
from upjab.vision.torchvision_plot import plot
from torchvision.io import read_image

# import torch
# torch.manual_seed(1)   

transforms_list = [v2.RandomCrop(size=(224, 224))]    

path = 'example_data/images/assets/astronaut.jpg'
transforms_list = [
    v2.RandomResizedCrop(size=(224, 224), antialias=True),
]

torchvision_transform_plot(path, transforms_list)

transform_functions = list(v2.__dict__.keys())

for i in transform_functions:
    try:
        transforms_list = [v2.__dict__[i]()]
        print(f'Transform: {i}')
        torchvision_transform_plot(path, transforms_list)
    except:
        print(f'FAIL Transform: {i}')
        # input()
        if i.startswith('_'):
            pass
        else:
            print(v2.__dict__[i].__doc__)
        # input()
        pass


from torchvision import tv_tensors  # we'll describe this a bit later, bare with us

img = read_image(path)

boxes = tv_tensors.BoundingBoxes(
    [
        [15, 10, 370, 510],
        [275, 340, 510, 510],
        [130, 345, 210, 425]
    ],
    format="XYXY", canvas_size=img.shape[-2:])

transforms = v2.Compose([
    v2.RandomResizedCrop(size=(224, 224), antialias=True),
    v2.RandomPhotometricDistort(p=1),
    v2.RandomHorizontalFlip(p=1),
])

out_img, out_boxes = transforms(img, boxes)
print(type(boxes), type(out_boxes))



plot([(img, boxes), (out_img, out_boxes)])



from upjab.tool.timer import timer
import time

with timer() as t:
    time.sleep(1.2345)
print(f'Print Elapsed time: {t.interval:>20.2f} seconds')



from experiment_configurations.v0001.args_setup import args_setup
from upjab.tool.args_setup_and_logger import args_setup_and_logger
args = args_setup("From_Experiment_Args_Setup")
logging = args_setup_and_logger(args)
logging.info(f'fake accuracy: 0.58 fake auc-roc: 0.88 fake loss: 0.42')
logging.info("done")
with open(args.checkpoint_save_path, 'w') as f:
    f.write('fake checkpoint')
print('')




import torch
from upjab.vision.i3d_video_feature import I3D_Video_Feature

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




from upjab.vision.read_video import read_video


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

from upjab.vision.read_image import read_image

path = 'example_data/images/disease_image/images_original/119799_objt_rs_2020-12-15_13-14-02-33_002.JPG'
image = read_image(path)

import numpy as np

image_np = np.array(image)



from upjab.tool.histogram_folder import histogram_folder


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

from upjab.tool.collect_to_one_folder import collect_to_one_folder

target_folder = 'example_data/text/white_histogram'
collect_to_one_folder(
    target_folder=target_folder,
    file_extends=['jpg', 'txt'],        
)


from upjab.tool.get_file_path_list import get_file_path_list
from upjab.tool.file_path_get_last_folder import file_path_get_last_folder


target_folder = 'example_data/images'
file_list = get_file_path_list(target_folder, file_extends=['jpg', 'JPG', 'json'])
for f_ in file_list:
    new_file_path = file_path_get_last_folder(f_, '_ALLGOOD')
    print(f_)
    print(new_file_path)
    print('--')


from upjab.tool.only_python_files import only_python_files

target_folder='example_data/images'
file_extends=['py']

only_python_files(
    target_folder=target_folder,
    file_extends=file_extends,
    
)

from upjab.tool.shuffle_split_folder import shuffle_split_folder

target_folder = 'example_data/text/white'
shuffle_split_folder(
    target_folder=target_folder,
    file_type_list=['txt'],
    random_seed=1,
    split_ratio=0.2
)

from upjab.tool.split_into_folders import split_into_folders
target_folder = 'example_data/text/black'
split_into_folders(
    target_folder=target_folder,
    file_extends=['jpg', 'txt'],
    cutting=3,
    is_shuffle=True,
    random_seed=5
)      


from upjab.tool.what_missing_in_folder import what_missing_in_folder
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


from upjab.tool.setup_logger import setup_logger

filepath = 'example_data/example.log'
logger = setup_logger(filepath)
logger.info('hi from upjab logger')