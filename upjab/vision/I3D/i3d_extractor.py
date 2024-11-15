try:
    from .i3d_model import I3D
    from .i3d_transform import i3d_transform_16frame
    # print("Using relative import")
except ImportError:
    from i3d_model import I3D
    from i3d_transform import i3d_transform_16frame
    # print("Using absoloute import")


import torch
import numpy as np


def get_slice(video_duration, target_number_segment=12, one_segment_size=16):
        
    end_point = video_duration - one_segment_size
    idx_float_list = np.arange(
        0, end_point, 
        step=end_point/(target_number_segment-1),
        dtype=float)
    idx_floor = np.floor(idx_float_list)
    idx_int = idx_floor.astype(np.int32)
    slice_start = np.append(idx_int, [end_point])
    return slice_start

class I3D_Extractor:
    def __init__(
        self,
        ckpt_path=None,        
        device=None,
        local_weight=False,
        ):
        self.i3d = I3D()
        self.i3d.eval()
        if device is None:
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.i3d.to(device)
        

        if ckpt_path is not None:
            print('Loading i3d model weight:')
            print(f'Path: {ckpt_path}')
            self.i3d.load_state_dict(torch.load(ckpt_path, map_location='cpu'))

        elif local_weight:
            import os
            weight_path = os.path.dirname(__file__) + '/i3d_rgb.pt'
            
            if os.path.isfile(weight_path):
                print('Loading local i3d model weight:')
                print(f'Path: {weight_path}')
                self.i3d.load_state_dict(torch.load(weight_path, map_location='cpu'))
            else:
                print('Local i3d model weight not found.')
                print(f'Path: {weight_path}')
                
        
        self.transform = i3d_transform_16frame  
   

    
    def to(self, device):
        self.i3d.to(device)


    def extract(self, numpy_16at15FPS_frames, is_ten_crop=True):
        device = next(self.i3d.parameters()).device
        torch_16_video_feature = self.transform(numpy_16at15FPS_frames, is_ten_crop=is_ten_crop)
        input_features = torch_16_video_feature.to(device)
        with torch.no_grad():
            output_tencrop = self.i3d.forward(input_features, features=True)
        return output_tencrop

    
    def extract_from_slice(self,
        THWC_rgb_numpy_video,
        is_ten_crop=True,
        target_number_segment=12,
        one_segment_size=16,
        return_numpy=True
        ):
        slice_start_idx = get_slice(
            video_duration=THWC_rgb_numpy_video.shape[0],
            target_number_segment=target_number_segment,
            one_segment_size=one_segment_size)
        
        output_list = []
        for startp in slice_start_idx:
            rgb_oneslice = THWC_rgb_numpy_video[startp:startp+one_segment_size,:,:,:]            
            
            with torch.no_grad():
                output_tencrop = self.extract(rgb_oneslice, is_ten_crop=is_ten_crop)

            output_list.append(output_tencrop)

        outputs = torch.stack(output_list)
        if return_numpy:
            numpy_outputs = outputs.cpu().numpy()
            return numpy_outputs
        else:
            return outputs
        
    



if __name__ == "__main__":
    # vid = torch.randint(0, 256, size=(105,360,480,3)).numpy()
    i3d = I3D_Extractor(local_weight=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    i3d.to(device)
    vid = np.random.randint(0, 256, size=(105,360,480,3))
    
    i3d_feature = i3d.extract_from_slice(
        THWC_rgb_numpy_video=vid,
        is_ten_crop=True,
        target_number_segment=11,
        return_numpy=False
    )

    i3d_feature = i3d.extract_from_slice(
        THWC_rgb_numpy_video=vid,
        is_ten_crop=True,
        target_number_segment=4,
        return_numpy=True
    )

    i3d_feature = i3d.extract_from_slice(
        THWC_rgb_numpy_video=vid,
        is_ten_crop=False,
        target_number_segment=11,
        return_numpy=True
    )
    print('end')