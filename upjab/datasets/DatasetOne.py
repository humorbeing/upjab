import torch
from typing import Any, Callable, Optional, Tuple
import glob

class DatasetOne(torch.utils.data.Dataset):    

    def __init__(
        self,
        root,
        data_loader: Callable[[str], Any],
        target_loader: Callable[[str], Any],
        extensions=['txt', 'jpg', 'JPG'],
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ) -> None:        
        
        file_list = []
        for ext_ in extensions:
            files = glob.glob(root + f'/**/*.{ext_}', recursive=True)        
            file_list.extend(files)

        self.data_loader = data_loader
        self.target_loader = target_loader
        self.samples = file_list
        self.transform=transform
        self.target_transform=target_transform

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path = self.samples[index]
        sample = self.data_loader(path)
        target = self.target_loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target

    def __len__(self) -> int:
        return len(self.samples)


if __name__ == '__main__':
    root = 'example_data/videos/fishes/crowd'
    def data_loader(path):
        return 1

    def target_loader(path):
        return 0
    d1 = DatasetOne(
        root,
        data_loader=data_loader,
        target_loader=target_loader,
        extensions=['mp4'],
        transform = None,
        target_transform = None,
    )


    print('done')