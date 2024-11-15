import torch
from torchvision.transforms import transforms
import torch.nn as nn


class ConvertBCHWtoCBHW(nn.Module):
    """Convert tensor from (B, C, H, W) to (C, B, H, W)"""

    def forward(self, vid: torch.Tensor) -> torch.Tensor:
        return vid.permute(1, 0, 2, 3)



class VideoClassificationPresetTrain:
    def __init__(
        self,
        *,
        crop_size,
        resize_size,
        mean=(0.43216, 0.394666, 0.37645),
        std=(0.22803, 0.22145, 0.216989),
        hflip_prob=0.5,
    ):
        trans = [
            transforms.ConvertImageDtype(torch.float32),
            # We hard-code antialias=False to preserve results after we changed
            # its default from None to True (see
            # https://github.com/pytorch/vision/pull/7160)
            # TODO: we could re-train the video models with antialias=True?
            transforms.Resize(resize_size, antialias=False),
        ]
        if hflip_prob > 0:
            trans.append(transforms.RandomHorizontalFlip(hflip_prob))
        trans.extend([transforms.Normalize(mean=mean, std=std), transforms.RandomCrop(crop_size), ConvertBCHWtoCBHW()])
        self.transforms = transforms.Compose(trans)

    def __call__(self, x):
        return self.transforms(x)


class VideoClassificationPresetEval:
    def __init__(self, *, crop_size, resize_size, mean=(0.43216, 0.394666, 0.37645), std=(0.22803, 0.22145, 0.216989)):
        self.transforms = transforms.Compose(
            [
                transforms.ConvertImageDtype(torch.float32),
                # We hard-code antialias=False to preserve results after we changed
                # its default from None to True (see
                # https://github.com/pytorch/vision/pull/7160)
                # TODO: we could re-train the video models with antialias=True?
                transforms.Resize(resize_size, antialias=False),
                transforms.Normalize(mean=mean, std=std),
                transforms.CenterCrop(crop_size),
                ConvertBCHWtoCBHW(),
            ]
        )

    def __call__(self, x):
        return self.transforms(x)


if __name__ == '__main__':

    video1 = torch.randint(0, 256, (8, 3, 360, 480))

    train_resize_size = (128, 171)
    train_crop_size = (112, 112)

    transform_train = VideoClassificationPresetTrain(        
        crop_size=train_crop_size,
        resize_size=train_resize_size,
        mean=(0.43216, 0.394666, 0.37645),
        std=(0.22803, 0.22145, 0.216989),
        hflip_prob=0.5)

    video2 = transform_train(video1)

    val_resize_size = (128, 171)
    val_crop_size = (112, 112)

    transform_test = VideoClassificationPresetEval(
        crop_size=val_crop_size,
        resize_size=val_resize_size,
        mean=(0.43216, 0.394666, 0.37645),
        std=(0.22803, 0.22145, 0.216989))
    
    video3 = transform_test(video1)
    print('done')