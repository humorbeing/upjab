from tqdm import tqdm
import json
import os
import glob
from PIL import Image 
import numpy as np

path = 'JsonCrop_dataset_annotations.json'

with open(path, 'r') as f:
    data = json.load(f)



target_folder = 'images_big_image_dataset/disease'

file_list = glob.glob(target_folder + '/**/*.JPG', recursive=True)
file_list2 = glob.glob(target_folder + '/**/*.jpg', recursive=True)
file_list.extend(file_list2)


problem_list = []
for fi in tqdm(file_list):
    try:
        file_name = fi.split('/')[-1]
        folder_path_list = fi.split('/')[:-1]
        if folder_path_list:
            pass
        else:
            folder_path_list = ['.']

        folder_path = os.path.join(*folder_path_list) + '_crop'
        if fi.startswith('/'):
            folder_path = f'/{folder_path}'
        os.makedirs(folder_path, exist_ok=True)

        image_info = data[file_name]
        bbox = image_info['annotation']['bbox']    
        x_cen, y_cen, width, height = bbox
        # x_min = int(x_cen - (width/2))
        # x_max = int(x_cen + (width/2))
        # y_min = int(y_cen - (height/2))
        # y_max = int(y_cen + (height/2))
        x_min = int(x_cen )
        x_max = int(x_cen + (width))
        y_min = int(y_cen )
        y_max = int(y_cen + (height))
        im = Image.open(fi)  
        img = np.array(im)
        max_y = img.shape[0]
        max_x = img.shape[1]
        x_bound = 0
        y_bound = 0
        y_min = y_min - y_bound
        y_max = y_max + y_bound
        x_min = x_min - x_bound
        x_max = x_max + x_bound
        if y_min < 0:
            y_min = 0
        if x_min < 0:
            x_min = 0

        if y_max > max_y:
            y_max = max_y
        if x_max > max_x:
            x_max = max_x
        crop = img[y_min:y_max, x_min:x_max, :]
        crop_img = Image.fromarray(crop, "RGB")
        save_name = folder_path + '/' + file_name
        
        crop_img.save(save_name)
        # break
    except:
        problem_list.append(fi)
    
    # break
    

if problem_list:
    with open('problems.txt', 'w') as f:
        for line in problem_list:
            f.writelines(line + '\n')
    
else:
    print('all good')
    


print('end')