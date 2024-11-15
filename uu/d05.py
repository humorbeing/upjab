
import numpy as np
import torch

from torch.utils.data import DataLoader
from torch.utils.data.sampler import BatchSampler


class BalancedBatchSampler(BatchSampler):
    """
    BatchSampler - from a MNIST-like dataset, samples n_classes and within these classes samples n_samples.
    Returns batches of size n_classes * n_samples
    """

    def __init__(self, dataset, n_classes, n_samples):
        loader = DataLoader(dataset)
        self.labels_list = []
        for _, label in loader:
            self.labels_list.append(label)
        self.labels = torch.LongTensor(self.labels_list)
        self.labels_set = list(set(self.labels.numpy()))
        self.label_to_indices = {label: np.where(self.labels.numpy() == label)[0]
                                 for label in self.labels_set}
        for l in self.labels_set:
            np.random.shuffle(self.label_to_indices[l])
        self.used_label_indices_count = {label: 0 for label in self.labels_set}
        self.count = 0
        self.n_classes = n_classes
        self.n_samples = n_samples
        self.dataset = dataset
        self.batch_size = self.n_samples * self.n_classes

    def __iter__(self):
        self.count = 0
        while self.count + self.batch_size <= len(self.dataset):
            classes = np.random.choice(self.labels_set, self.n_classes, replace=False)
            indices = []
            for class_ in classes:
                indices.extend(self.label_to_indices[class_][
                               self.used_label_indices_count[class_]:self.used_label_indices_count[
                                                                         class_] + self.n_samples])
                self.used_label_indices_count[class_] += self.n_samples
                if self.used_label_indices_count[class_] + self.n_samples > len(self.label_to_indices[class_]):
                    np.random.shuffle(self.label_to_indices[class_])
                    self.used_label_indices_count[class_] = 0
            yield indices
            self.count += self.n_classes * self.n_samples

    def __len__(self):
        return len(self.dataset) // self.batch_size
    


import torchvision

dataset_root = 'example_data/text/long-tail_dataset'
extensions=['.txt']
def whatis(x):
    return 0
temp11 = torchvision.datasets.DatasetFolder(root=dataset_root, loader=whatis, extensions=extensions, transform=None)



def get_class_distribution(dataset_obj):
    count_dict = {k:0 for k,v in dataset_obj.class_to_idx.items()}
    idx2class = {v: k for k, v in dataset_obj.class_to_idx.items()}
    for element in dataset_obj:
        y_lbl = element[1]
        y_lbl = idx2class[y_lbl]
        count_dict[y_lbl] += 1
            
    return count_dict
print("Distribution of classes: \n", get_class_distribution(temp11))

import torch
target_list = torch.tensor(temp11.targets)
class_count = [i for i in get_class_distribution(temp11).values()]
class_weights = 1./torch.tensor(class_count, dtype=torch.float) 

class_weights_all = class_weights[target_list]

from torch.utils.data import WeightedRandomSampler, DataLoader
# from sampler import RASampler
# train_sampler = RASampler(temp11, shuffle=True, repetitions=3)
        
# train_sampler = torch.utils.data.distributed.DistributedSampler(temp11)
balanced_batch_sampler = BalancedBatchSampler(temp11, 2, 3)
train_loader = DataLoader(dataset=temp11, batch_sampler=balanced_batch_sampler)

for batch in train_loader:
    print(batch)


temp21 = iter(train_loader)
temp22 = next(temp21)

print('done')