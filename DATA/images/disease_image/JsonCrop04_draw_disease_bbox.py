from tqdm import tqdm
import json
import cv2
import glob
import os


path = 'JsonCrop_dataset_annotations.json'

with open(path, 'r') as f:
    data = json.load(f)


target_folder = 'images_big_image_dataset/disease'

file_list = glob.glob(target_folder + '/**/*.JPG', recursive=True)
file_list2 = glob.glob(target_folder + '/**/*.jpg', recursive=True)
file_list.extend(file_list2)

image_names = {}

for f_ in file_list:
    f_name = f_.split('/')[-1]
    image_names[f_name] = f_



disease_counter = 0
for img in tqdm(image_names):
    image_path = image_names[img]
    folder_path_list = image_path.split('/')[:-1]
    if folder_path_list:
        pass
    else:
        folder_path_list = ['.']

    folder_path = os.path.join(*folder_path_list) + '_bbox'
    if image_path.startswith('/'):
        folder_path = f'/{folder_path}'
    os.makedirs(folder_path, exist_ok=True)


    bgr = cv2.imread(image_path)
    is_disease = data[img]['annotation']['diseases_exist']
    if is_disease:
        
        disease_bbox = data[img]['annotation']['diseases_bbox']
        
        [xmin, ymin, width, height] = disease_bbox
        xmax = xmin + width
        ymax = ymin + height
        cv2.rectangle(bgr, (int(xmin), int(ymin)), (int(xmax), int(ymax)), color=(255,255,0), thickness=3)

        # category_str = data[img]['annotation']['diseases_desc']
        # cv2.putText(rgb, category_str, (xmin, ymin-5), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 3, cv2.LINE_AA) 
    disease_bbox = data[img]['annotation']['bbox']
    [xmin, ymin, width, height] = disease_bbox
    xmax = xmin + width
    ymax = ymin + height
    cv2.rectangle(bgr, (int(xmin), int(ymin)), (int(xmax), int(ymax)), color=(255,255,0), thickness=3)
    
    save_name = folder_path + '/' + img
    cv2.imwrite(save_name, bgr)
    # print()

print('')