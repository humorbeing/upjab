


import glob
from tool_video_extractor import video_to_i3d_numpy
import os
import numpy as np
from tqdm import tqdm


def video_folder_to_i3d(
    target_folder,    
    is_cv2=True,
    feeding_fps=15,
    is_ten_crop=True,
    target_number_segment=12,
    return_numpy=True
    ):

    file_list = glob.glob(target_folder + '/**/*.mp4', recursive=True)

    for f_ in tqdm(file_list):
        i3d_feature = video_to_i3d_numpy(
            video_path=f_,
            is_cv2=is_cv2,
            feeding_fps=feeding_fps,
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
        # print(f'save {save_path3}')


if __name__ == '__main__':
    target_folder = 'dataset_and_weight/videos/fishes/crowd'

    video_folder_to_i3d(
        target_folder=target_folder,    
        is_cv2=True,
        feeding_fps=15,
        is_ten_crop=True,
        target_number_segment=12,
        return_numpy=True
    )

    target_folder = 'dataset_and_weight/videos/fishes/not_crowd'

    video_folder_to_i3d(
        target_folder=target_folder,    
        is_cv2=True,
        feeding_fps=15,
        is_ten_crop=True,
        target_number_segment=12,
        return_numpy=True
    )
    print('end')
