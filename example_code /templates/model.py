import torch
import torch.nn as nn


def transform(x):
    return x

class ModelName(nn.Module):
    def __init__(self):
        super(ModelName, self).__init__()

        
        self.transform_train_data = transform
        self.transform_train_label = None

        self.transform_eval_data = transform        
        self.transform_eval_label = None

        self.transform_inference_data = transform

        pass

    def forward(self, x):        
        return self.forward_train_batch(x)
    
    def forward_train_batch(self, x):
        return 0
    
    def forward_eval_batch(self, x):
        return 0
    
    def forward_inference(self, x):
        return 0


if __name__ == '__main__':
    model = ModelName()
    input_batch = torch.randn(2, 3, 224, 224)
    label_batch = torch.randint(0, 10, (2,))

    batch_transformed = model.transform_train_data(input_batch)

    output = model.forward_train_batch(batch_transformed)
    print('done')
    
