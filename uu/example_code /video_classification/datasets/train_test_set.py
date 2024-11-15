from upjab.datasets.DatasetOne import DatasetOne
from upjab.vision.VideoClips import VideoClips
from numpy.random import randint
import torch
from upjab.transforms.video_presets import VideoClassificationPresetTrain, VideoClassificationPresetEval


def train_random_loader(path, clip_num=1):
    vc = VideoClips(path)
    _r = len(vc)
    assert clip_num <= _r, 'clip_num must be less than or equal to the number of clips in the video'
    if clip_num == 1:
        return vc.get_clip(randint(0, _r))[0]
    else:
        idx = randint(0, _r, size=clip_num)
        results = []
        for i in idx:
            results.append(vc.get_clip(i)[0])
        
        return torch.stack(results)
    

def target_loader(path):
    check = '/crowd/'
    if check in path:
        return 1
    else:
        return 0

train_resize_size = (128, 171)
train_crop_size = (112, 112)

transform_train = VideoClassificationPresetTrain(        
    crop_size=train_crop_size,
    resize_size=train_resize_size,
    mean=(0.43216, 0.394666, 0.37645),
    std=(0.22803, 0.22145, 0.216989),
    hflip_prob=0.5)


transform_test = VideoClassificationPresetEval(
    crop_size=train_crop_size,
    resize_size=train_resize_size,
    mean=(0.43216, 0.394666, 0.37645),
    std=(0.22803, 0.22145, 0.216989))

root = 'example_data/videos/fishes/crowd'

crowd_dataset = DatasetOne(
    root,
    data_loader=train_random_loader,
    target_loader=target_loader,
    extensions=['mp4'],
    transform = transform_train,
    target_transform = None,
)

# temp11 = iter(crowd_dataset)
# temp12 = next(temp11)

root = 'example_data/videos/fishes/not_crowd'

not_crowd_dataset = DatasetOne(
    root,
    data_loader=train_random_loader,
    target_loader=target_loader,
    extensions=['mp4'],
    transform = transform_train,
    target_transform = None,
)


# temp11 = iter(not_crowd_dataset)
# temp12 = next(temp11)


root = 'example_data/videos/fishes'

test_dataset = DatasetOne(
    root,
    data_loader=train_random_loader,
    target_loader=target_loader,
    extensions=['mp4'],
    transform = transform_test,
    target_transform = None,
)


# temp11 = iter(test_dataset)
# temp12 = next(temp11)

# print('done')