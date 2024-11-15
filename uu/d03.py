import torchvision
from upjab.datasets.get_class_distribution import get_class_distribution


dataset_root = 'example_data/text/long-tail_dataset'

def whatis(x):
    return 0
temp11 = torchvision.datasets.DatasetFolder(root=dataset_root, loader=whatis, extensions=['.txt'], transform=None)


print("Distribution of classes: \n", get_class_distribution(temp11))

import torch
target_list = torch.tensor(temp11.targets)
class_count = [i for i in get_class_distribution(temp11).values()]
class_weights = 1./torch.tensor(class_count, dtype=torch.float) 

class_weights_all = class_weights[target_list]

from torch.utils.data import WeightedRandomSampler, DataLoader
weighted_sampler = WeightedRandomSampler(
    weights=class_weights_all,
    num_samples=len(class_weights_all),
    replacement=True
)

train_loader = DataLoader(dataset=temp11, shuffle=False, batch_size=2, sampler=weighted_sampler)

for batch in train_loader:
    print(batch)

print('done')