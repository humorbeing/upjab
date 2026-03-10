# sudo apt install graphviz
# pip install torchview


from torchview import draw_graph
import torch
from torchvision.models import resnet50    

model = resnet50()
# x = torch.rand(1, 3, 224, 224)
model_graph = draw_graph(
    model,
    input_size=(1,3,224,224),
    expand_nested=True,
    hide_inner_tensors=False,
    hide_module_functions=False,
    roll=True,
    # save_graph=True,
    # filename="resnet50_torchview1.svg",
    )
# model_graph.visual_graph
from upjab import AP
# import os
# os.makedirs(AP("saves"),exist_ok=True)
model_graph.visual_graph.render(AP("saves/resnet50_torchview"), format="svg")
model_graph.visual_graph.render(AP("saves/resnet50_torchview"), format="png")