
import os
import glob
from tqdm import tqdm
import shutil

def histogram_folder(
    target_folder,
    detect,
    file_type_list=['JPG','jpg']
    ):

    save_path = target_folder + '_histogram'
    for i in range(10):
        new_path = f'{save_path}/{i}'
        os.makedirs(new_path, exist_ok=True)

    file_list = []
    for ty_ in file_type_list:
        temp_file_list = glob.glob(target_folder + f'/**/*.{ty_}', recursive=True)    
        file_list.extend(temp_file_list)
    

    for video_path in tqdm(file_list):
        score = detect(video_path)
        if score == 1:
            cata = '9'
        else:
            str_score = str(score)
            cata = str_score[2]
        shutil.copy(video_path, f'{save_path}/{cata}')
        


if __name__ == '__main__':
    from N05_detect import detect
    root = os.path.dirname(__file__)
    target_folder = f'{root}/data/color/03_randomsample/labeled'
    
    histogram_folder(
        target_folder=target_folder,
        detect=detect,
        file_type_list=['png']
    )

    target_folder = f'{root}/data/color/03_randomsample/unlabeled'
    
    histogram_folder(
        target_folder=target_folder,
        detect=detect,
        file_type_list=['png']
    )

    target_folder = f'{root}/data/color/02_dataset'
    
    histogram_folder(
        target_folder=target_folder,
        detect=detect,
        file_type_list=['png']
    )

    # Then
    # Create 04_histogram folder
    # Move 02_dataset_histogram to 04_histogram
    # Move labeled_histogram to 04_histogram
    # Move unlabeled_histogram to 04_histogram


    print('end')