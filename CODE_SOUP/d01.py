# https://pytorch.org/vision/stable/generated/torchvision.datasets.DatasetFolder.html#torchvision.datasets.DatasetFolder

import torchvision

dataset_root = 'example_data/text/long-tail_dataset'

def whatis(x):
    return 0
temp11 = torchvision.datasets.DatasetFolder(root=dataset_root, loader=whatis, extensions=['.txt'], transform=None)

temp12 = iter(temp11)
temp13 = next(temp12)


for i in temp11:
    print(i)


print(temp11.classes)
print(temp11.class_to_idx)
import torch


temp21 = torch.utils.data.DataLoader(temp11, batch_size=2, shuffle=True, num_workers=0)

temp22 = iter(temp21)
temp23 = next(temp22)

from torch.utils.data import RandomSampler

sampler = RandomSampler(temp11)
temp31 = torch.utils.data.DataLoader(temp11, batch_size=4, sampler=sampler)

temp32 = iter(temp31)
temp33 = next(temp32)

print('done')