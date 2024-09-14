video_path = 'example_data/videos/fishes/crowd/00000001.mp4'


try:
    from .read_video import read_video   
    from .I3D.i3d_extractor import I3D_Extractor
    # print("Using relative import")
except ImportError:
    from read_video import read_video
    from I3D.i3d_extractor import I3D_Extractor
    # print("Using absoloute import")


class I3D_Video_Feature:
    def __init__(
        self,
        ckpt_path=None,        
        device=None,
        local_weight=False):

        self.video_reader = read_video
        self.i3d_extractor = I3D_Extractor(
            ckpt_path=ckpt_path,
            device=device,
            local_weight=local_weight)
    
    def extract_video(
        self,
        video_path,
        is_ten_crop: bool = True,
        target_number_segment: int = 12,
        # one_segment_size: int = 16,
        return_numpy: bool = True):
        
        video_np = self.video_reader(
            video_path=video_path,
            is_cv2=True,
            feeding_fps=15,
            echo=False)
        
        extracted_feature = self.i3d_extractor.extract_from_slice(
            THWC_rgb_numpy_video=video_np,
            is_ten_crop=is_ten_crop,
            target_number_segment=target_number_segment,
            # one_segment_size=one_segment_size,
            return_numpy=return_numpy)
        return extracted_feature


    def extract_folder(
        self,
        folder_path,
        file_extends=['mp4', 'avi'],
        is_ten_crop: bool = True,
        target_number_segment: int = 12,        
        return_numpy: bool = True):

        import glob
        from tqdm import tqdm
        import numpy as np
        import os

        file_list = []
        for ext_ in file_extends:
            files = glob.glob(folder_path + f'/**/*.{ext_}', recursive=True)        
            file_list.extend(files)
        
        for f_ in tqdm(file_list):
            i3d_feature = self.extract_video(
                video_path=f_,                
                is_ten_crop=is_ten_crop,
                target_number_segment=target_number_segment,
                return_numpy=return_numpy   
                )
            
            file_name = f_.split('/')[-1][:-4]
            folder_name = f_.split('/')[-2]
            
            save_path1 = os.path.join(*f_.split('/')[:-2])
            if f_.startswith('/'):
                save_path1 = f'/{save_path1}'
            save_path2 = f'{save_path1}/{folder_name}_i3d'
            save_path3 = f'{save_path2}/{file_name}.npy'
            os.makedirs(save_path2, exist_ok=True)
            np.save(save_path3, i3d_feature)


if __name__ == '__main__':
    import torch
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


    ckpt_path = None
    i3d_video_feature = I3D_Video_Feature(
        ckpt_path=ckpt_path,        
        device=device,
        local_weight=False)

    video_path = 'example_data/videos/fishes/crowd/00000001.mp4'
    video_feature = i3d_video_feature.extract_video(
        video_path=video_path,
        is_ten_crop=True,
        target_number_segment=12,        
        return_numpy=True)
    
    print(video_feature.shape)

    video_feature = i3d_video_feature.extract_video(
        video_path=video_path,
        is_ten_crop=False,
        target_number_segment=7,        
        return_numpy=False)
    
    print(video_feature.shape)

    video_feature = i3d_video_feature.extract_video(
        video_path=video_path,
        is_ten_crop=True,
        target_number_segment=4,        
        return_numpy=False)

    print(video_feature.shape)

    folder_path = 'example_data/videos/fishes/not_crowd'
    i3d_video_feature.extract_folder(folder_path=folder_path)


    print('end')