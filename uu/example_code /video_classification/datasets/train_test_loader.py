try:
    from .train_test_set import crowd_dataset
    from .train_test_set import not_crowd_dataset
    from .train_test_set import test_dataset
    print("Using relative import")
except ImportError:
    from train_test_set import crowd_dataset
    from train_test_set import not_crowd_dataset
    from train_test_set import test_dataset
    print("Using absoloute import")




from torch.utils.data import DataLoader

def get_dataset_loader(batch_size, is_max_out=True):
    abtrain_dataset = crowd_dataset
    ntrain_dataset = not_crowd_dataset
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
        'train_batch_size': train_batch_size,
        'test_batch_size': batch_size
    }






if __name__ == '__main__':
    loaders = get_dataset_loader(50)
    print('end')