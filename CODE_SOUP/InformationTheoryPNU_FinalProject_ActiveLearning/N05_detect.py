# from N03_model import Net
import torch
import os
from torch.nn.functional import softmax

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# model = get_disease_detection_model().to(device).eval()
# model = Net()
from N04_resnet import ResNet18
model = ResNet18()
num_features = model.linear.in_features #the number of nodes in the last layer for the ResNet model
num_classes = 2 #whatever number of classes u have.
model.linear = torch.nn.Linear(num_features, num_classes) 


root = os.path.dirname(__file__)
model_ckpt = f'{root}/model_weights/labeled_0.704_model.pkl'
model.load_state_dict(torch.load(model_ckpt, map_location='cpu'))

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'device: {device}')
model = model.to(device)
model.eval()

from torchvision import transforms
transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
        ])

from PIL import Image


def detect(img_path):
    # bgr_frame = cv2.imread(img_path)
    # im = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
    im = Image.open(img_path)
    images = transform(im)
    # images = transform_disease_detection(im)
    images = images[None, :]
    images = images.to(device)
    with torch.no_grad():
        temp12 = model(images)
        
        sf = softmax(temp12, dim=1)        
        prediction = sf[:,1].item()

    return prediction

if __name__ == '__main__':        
    
    img_path = 'Mark99/data/color/00_start_dataset/0/00000030.png'
    score = detect(img_path)
    print(score)
