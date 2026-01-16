# pip install tensorboard
import torch
from torchvision.models import resnet50
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter("saves/torchlogs/")
model = resnet50()
x = torch.rand(1, 3, 224, 224)    
writer.add_graph(model, x)
writer.close()

# tensorboard --logdir=./saves/torchlogs
