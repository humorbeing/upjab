import numpy as np
import torch
from torchvision.transforms import Resize
resize = Resize(256)
from torchvision.transforms import TenCrop
tencrop = TenCrop([224,224])
from torchvision.transforms import CenterCrop
centercrop = CenterCrop(224)



def i3d_transform_16frame(THWC_rgb_numpy_video, is_ten_crop=True):
    rgb = torch.from_numpy(THWC_rgb_numpy_video)

    # THWC to TCHW. TCHW is pytorch prefered dimension order
    # T here is regard as Batch-size. for example, normalization
    rgb_permute_TCHW = torch.permute(rgb, [0,3,1,2])


    # resize to 256 for cropping.
    # resize here, to save computation
    
    rgb_resize = resize.forward(rgb_permute_TCHW)

    # 255 to -1 to 1
    rgb_dev255 = (rgb_resize * 2 / 255) - 1.


    #############################  10 crop and one crop ###################
    # # training, 10 crop
    if is_ten_crop:    
        rgb_temp11 = tencrop.forward(rgb_dev255)
        rgb_tencrop = torch.stack(rgb_temp11)

    # inference, one crop
    else:
        rgb_center_crop = centercrop.forward(rgb_dev255)
        rgb_onecrop = rgb_center_crop[None, ...]
        rgb_tencrop = rgb_onecrop  # just reuse the code
    #############################  10 crop and one crop ###################


    # Ncrop T C H W   to   Ncrop C T H W
    NcCTHW_torch_i3d_video = torch.permute(rgb_tencrop, [0,2,1,3,4])
    return NcCTHW_torch_i3d_video


if __name__ == "__main__":
    
    # vid = torch.randint(0, 256, size=(105,360,480,3)).numpy()
    vid = np.random.randint(0, 256, size=(16,360,480,3))
    NcCTHW_torch_i3d_video = i3d_transform_16frame(vid, is_ten_crop=True)
    # NcCTHW_torch_i3d_video = torch.rand(10,3,105,224,224)
    
    print('end')