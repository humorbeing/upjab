import torch
import math
import random



class RandomErasing(object):
    '''
    Class that performs Random Erasing in Random Erasing Data Augmentation by Zhong et al.
    -------------------------------------------------------------------------------------
    probability: The probability that the operation will be performed.
    sl: min erasing area
    sh: max erasing area
    r1: min aspect ratio
    mean: erasing value
    random_fill: If ture, fill the erased area with random number. If false: fill with image net mean.
    -------------------------------------------------------------------------------------
    '''

    def __init__(self, probability=0.5, sl=0.02, sh=0.2, r1=0.3, mean=(0., 0., 0.), random_fill=False):
        self.probability = probability
        self.mean = mean
        self.sl = sl
        self.sh = sh
        self.r1 = r1
        self.random_fill=random_fill

    def __call__(self, img):

        if random.uniform(0, 1) > self.probability:
            return img

        for attempt in range(100):
            area = img.size()[1] * img.size()[2]

            target_area = random.uniform(self.sl, self.sh) * area
            aspect_ratio = random.uniform(self.r1, 1 / self.r1)

            h = int(round(math.sqrt(target_area * aspect_ratio)))
            w = int(round(math.sqrt(target_area / aspect_ratio)))

            if w <= img.size()[2] and h <= img.size()[1]:
                x1 = random.randint(0, img.size()[1] - h)
                y1 = random.randint(0, img.size()[2] - w)
                if not self.random_fill:
                    if img.size()[0] == 3:
                        img[0, x1:x1 + h, y1:y1 + w] = self.mean[0]
                        img[1, x1:x1 + h, y1:y1 + w] = self.mean[1]
                        img[2, x1:x1 + h, y1:y1 + w] = self.mean[2]
                    else:
                        img[0, x1:x1 + h, y1:y1 + w] = self.mean[0]
                else:
                    if img.size()[0] == 3:
                        img[0, x1:x1 + h, y1:y1 + w] = torch.randn((h, w))
                        img[1, x1:x1 + h, y1:y1 + w] = torch.randn((h, w))
                        img[2, x1:x1 + h, y1:y1 + w] = torch.randn((h, w))
                    else:
                        img[0, x1:x1 + h, y1:y1 + w] = torch.rand((h, w))
                return img

        return img



if __name__ == '__main__':
    import numpy as np
    img_np_int = np.random.randint(0, 256, size=(244, 244, 3), dtype=np.uint8)
    img_np_float = img_np_int / 255

    from torchvision import transforms
    toPIL = transforms.ToPILImage()
    # from torch import functional as F
    # from torch.nn import functional as F
    from torchvision.transforms import functional as F
    
    try: 
        img = F.to_pil_image(img_np_int)
    except:
        print('numpy int cannot convert to PIL image. use float')
    img_pil = F.to_pil_image(img_np_float)
    img_pil = toPIL(img_np_float)
    
    img_torch_int = torch.randint(0, 256, size=(244, 244, 3))
    try: 
        img = F.to_pil_image(img_torch_int)
        print('torch int convert to PIL image.')
    except:
        print('torch int cannot convert to PIL image. use float')

    img_torch_float = torch.rand(size=(244, 244, 3))
    img_torch_float = torch.rand(size=(3, 244, 244))
    try: 
        img = F.to_pil_image(img_torch_float)
        print('torch float convert to PIL image.')
    except:
        print('torch float cannot convert to PIL image.')



    from torchvision import transforms
    from torchvision.transforms.functional import InterpolationMode

    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD = [0.229, 0.224, 0.225]
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    scale_image_size = [int(x * 1.125) for x in [224,224]]

    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomResizedCrop(224, scale=(0.7, 1.0), interpolation=InterpolationMode.BICUBIC),
        transforms.Resize((112, 112)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        transforms.Resize((256,256)),         
        transforms.CenterCrop(224),
        transforms.Resize(scale_image_size),
        transforms.RandomCrop([224,224]),
        transforms.RandomHorizontalFlip(),
    ])

    
    from PIL import Image    
      
    im2arr = np.array(img_pil) # im2arr.shape: height x width x channel
    img_pil_again = Image.fromarray(im2arr)
    
    image_path = 'dataset/MARS/bbox_train/bbox_train/0073/0073C1T0001F029.jpg'
    try:
        img = Image.open(image_path).convert('RGB')
    except:
        img = img_pil_again
    transform = transforms.Compose([        
        transforms.ToTensor(),        
    ])
    img_torch = transforms.ToTensor()(img)
    img_torch = transform(img)

    randomerasing = RandomErasing(random_fill=True)
    img_torch_randomerasing = randomerasing(img_torch)
    img_randomerasing = F.to_pil_image(img_torch_randomerasing)
    print('End')