# try:
#     from .dataset_red_seabream_crowd import Dataset_abnormal
#     from .dataset_red_seabream_crowd import Dataset_normal
#     from .dataset_red_seabream_crowd import Dataset_test
#     print("Using relative import")
# except ImportError:
#     from dataset_red_seabream_crowd import Dataset_abnormal
#     from dataset_red_seabream_crowd import Dataset_normal
#     from dataset_red_seabream_crowd import Dataset_test
#     print("Using absoloute import")

from N04_dataset import Dataset_normal, Dataset_abnormal, Dataset_test


from torch.utils.data import DataLoader

def get_dataset_loader(batch_size, labeled_path=None, is_max_out=True):
    abtrain_dataset = Dataset_abnormal(labeled_path)
    ntrain_dataset = Dataset_normal(labeled_path)
    if is_max_out:
        temp11 = len(abtrain_dataset)
        temp12 = len(ntrain_dataset)
        if temp11 < temp12:
            train_batch_size = temp11
        else:
            train_batch_size = temp12

        if train_batch_size > batch_size:
            train_batch_size = batch_size
    
    else:
        train_batch_size = batch_size

    print('Train loader')
    print(f'number of abnormal samples: {len(abtrain_dataset)}')
    print(f'number of normal samples: {len(ntrain_dataset)}')
    print(f'train batch size: {train_batch_size}')

    abtrain_loader = DataLoader(
        abtrain_dataset,
        batch_size=train_batch_size, shuffle=True,
        num_workers=0, pin_memory=False, drop_last=True)

    ntrain_loader = DataLoader(
        ntrain_dataset,
        batch_size=train_batch_size, shuffle=True,
        num_workers=0, pin_memory=False, drop_last=True)

    test_dataset = Dataset_test()
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size, shuffle=False,
        num_workers=0, pin_memory=False, drop_last=False)
    
    print('Test loader')
    print(f'number of test samples: {len(test_dataset)}')
    print(f'test batch size: {batch_size}')

    return {
        'train_normal_loader': ntrain_loader,
        'train_abnormal_loader': abtrain_loader,
        'test_loader': test_loader,
    }



if __name__ == '__main__':
    import os
    root = os.path.dirname(__file__)
    labeled_path = f'{root}/data/color/03_randomsample/labeled'
    loaders = get_dataset_loader(50, labeled_path=labeled_path)
    print('end')