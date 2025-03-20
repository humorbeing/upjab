import torch.utils.data as data
import os
import numpy as np
import glob

from PIL import Image
from torchvision import transforms
# from torchvision.transforms.functional import InterpolationMode

root = os.path.dirname(__file__)


transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

class Dataset_normal(data.Dataset):
    def __init__(self, labeled_path=f'{root}/data/color/03_randomsample/labeled'):
        target_folder = labeled_path + '/normal'
        file_list = glob.glob(target_folder + '/**/*.png', recursive=True)
        
        # print(target_folder)
        self.list = file_list
        

        self.transform = transform
    def __len__(self):
        return len(self.list)
    
    def __getitem__(self, index):
        label = 0
        # im = Image.open(self.list[index]).convert('RGB')
        im = Image.open(self.list[index])
        img = self.transform(im)
        return img, label


class Dataset_abnormal(data.Dataset):
    def __init__(self, labeled_path=f'{root}/data/color/03_randomsample/labeled'):
        target_folder = labeled_path + '/abnormal'
        file_list = glob.glob(target_folder + '/**/*.png', recursive=True)
        
        # print(target_folder)
        self.list = file_list
        

        self.transform = transform
    def __len__(self):
        return len(self.list)
    
    def __getitem__(self, index):
        label = 1
        im = Image.open(self.list[index])
        img = self.transform(im)
        return img, label




class Dataset_test(data.Dataset):
    def __init__(self, DATASET_PATH = f'{root}/data/color/02_dataset'):
        target_folder = DATASET_PATH
        file_list = glob.glob(target_folder + '/**/*.png', recursive=True)
        
        self.list = file_list
        

        self.transform = transform
    def __len__(self):
        return len(self.list)
    
    def __getitem__(self, index):
        image_path = self.list[index]
        check_str = '/abnormal/'
        if check_str in image_path:
            label = 1
        else:
            label = 0
        im = Image.open(self.list[index])
        img = self.transform(im)
        return img, label



     

if __name__ == '__main__':
    from torch.utils.data import DataLoader    
    ntrain_dataset = Dataset_normal()
    temp11 = iter(ntrain_dataset)
    get_ntrain = next(temp11)

    ntrain_loader = DataLoader(
        ntrain_dataset,
        batch_size=16, shuffle=True,
        num_workers=0, pin_memory=False, drop_last=True)
    
    temp11 = iter(ntrain_loader)
    get_ntrain = next(temp11)

    abtrain_dataset = Dataset_abnormal()
    temp12 = iter(abtrain_dataset)
    get_abtrain = next(temp12)
    abtrain_loader = DataLoader(
        abtrain_dataset,
        batch_size=2, shuffle=True,
        num_workers=0, pin_memory=False, drop_last=False)

    temp12 = iter(abtrain_loader)
    get_abtrain = next(temp12)
    
    test_dataset = Dataset_test()
    temp12 = iter(test_dataset)
    get_abtrain = next(temp12)
    all_loader = DataLoader(
        test_dataset,
        batch_size=16, shuffle=False,
        num_workers=0, pin_memory=False, drop_last=False)

    temp12 = iter(all_loader)
    get_abtrain = next(temp12)

    print('end')
