import torch
from torch.utils.data import Dataset, RandomSampler

class TestDataSet(Dataset):
    def __len__(self):
        return 10
    
    def __getitem__(self, idx):
        return {"input":torch.tensor([idx, 2*idx, 3*idx], dtype=torch.float32), 
                "label": torch.tensor(idx, dtype=torch.float32)}

test_dataset = TestDataSet()

dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=4)

for data in dataloader:
    print(data)


for data in dataloader:
    print(data['input'].shape, data['label'])



test_dataset = TestDataSet()
sampler = RandomSampler(test_dataset)
dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=4, sampler=sampler)

for data in dataloader:
    print(data)


print('done')