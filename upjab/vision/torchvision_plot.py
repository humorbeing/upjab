import matplotlib.pyplot as plt
import torch
from torchvision.utils import draw_bounding_boxes, draw_segmentation_masks
from torchvision import tv_tensors
from torchvision.transforms.v2 import functional as F

from torchvision.transforms import v2
from torchvision.io import read_image


def plot(imgs, row_title=None, show=True, **imshow_kwargs):
    if not isinstance(imgs[0], list):
        # Make a 2d grid even if there's just 1 row
        imgs = [imgs]

    num_rows = len(imgs)
    num_cols = len(imgs[0])
    _, axs = plt.subplots(nrows=num_rows, ncols=num_cols, squeeze=False)
    for row_idx, row in enumerate(imgs):
        for col_idx, img in enumerate(row):
            boxes = None
            masks = None
            if isinstance(img, tuple):
                img, target = img
                if isinstance(target, dict):
                    boxes = target.get("boxes")
                    masks = target.get("masks")
                elif isinstance(target, tv_tensors.BoundingBoxes):
                    boxes = target
                else:
                    raise ValueError(f"Unexpected target type: {type(target)}")
            img = F.to_image(img)
            if img.dtype.is_floating_point and img.min() < 0:
                # Poor man's re-normalization for the colors to be OK-ish. This
                # is useful for images coming out of Normalize()
                img -= img.min()
                img /= img.max()

            img = F.to_dtype(img, torch.uint8, scale=True)
            if boxes is not None:
                img = draw_bounding_boxes(img, boxes, colors="yellow", width=3)
            if masks is not None:
                img = draw_segmentation_masks(img, masks.to(torch.bool), colors=["green"] * masks.shape[0], alpha=.65)

            ax = axs[row_idx, col_idx]
            ax.imshow(img.permute(1, 2, 0).numpy(), **imshow_kwargs)
            ax.set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])

    if row_title is not None:
        for row_idx in range(num_rows):
            axs[row_idx, 0].set(ylabel=row_title[row_idx])

    plt.tight_layout()
    if show:
        plt.show()



def torchvision_transform_plot(PATH, transforms_list):
    img = read_image(PATH)
    transforms = v2.Compose(transforms_list)
    out = transforms(img)
    plot([img, out])
    



if __name__ == "__main__":    
    
    torch.manual_seed(1)   

    transforms_list = [v2.RandomCrop(size=(224, 224))]    

    path = 'example_data/images/assets/astronaut.jpg'
    transforms_list = [
        v2.RandomResizedCrop(size=(224, 224), antialias=True),
    ]

    torchvision_transform_plot(path, transforms_list)

    transform_functions = list(v2.__dict__.keys())

    for i in transform_functions:
        try:
            transforms_list = [v2.__dict__[i]()]
            print(f'Transform: {i}')
            torchvision_transform_plot(path, transforms_list)
        except:
            print(f'FAIL Transform: {i}')
            # input()
            if i.startswith('_'):
                pass
            else:
                print(v2.__dict__[i].__doc__)
            # input()
            pass


    from torchvision import tv_tensors  # we'll describe this a bit later, bare with us

    img = read_image(path)

    boxes = tv_tensors.BoundingBoxes(
        [
            [15, 10, 370, 510],
            [275, 340, 510, 510],
            [130, 345, 210, 425]
        ],
        format="XYXY", canvas_size=img.shape[-2:])

    transforms = v2.Compose([
        v2.RandomResizedCrop(size=(224, 224), antialias=True),
        v2.RandomPhotometricDistort(p=1),
        v2.RandomHorizontalFlip(p=1),
    ])

    out_img, out_boxes = transforms(img, boxes)
    print(type(boxes), type(out_boxes))

    

    plot([(img, boxes), (out_img, out_boxes)])

    
    