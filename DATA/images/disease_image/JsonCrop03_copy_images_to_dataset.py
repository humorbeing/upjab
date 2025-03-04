import json

path = 'JsonCrop_dataset_annotations.json'

with open(path, 'r') as f:
    data = json.load(f)


import glob
target_folder = 'images_original'
file_list = glob.glob(target_folder + '/**/*.JPG', recursive=True)
file_list2 = glob.glob(target_folder + '/**/*.jpg', recursive=True)
file_list.extend(file_list2)

image_names = {}

for f_ in file_list:
    f_name = f_.split('/')[-1]
    image_names[f_name] = f_


from tqdm import tqdm
import shutil
import os

disease_folder = 'images_big_image_dataset/disease'
normal_folder = 'images_big_image_dataset/normal'

os.makedirs(disease_folder, exist_ok=True)
os.makedirs(normal_folder, exist_ok=True)

disease_counter = 0
normal_counter = 0

for img in tqdm(data):
    is_disease = data[img]['annotation']['diseases_exist']
    if is_disease:
        shutil.copy(image_names[img], disease_folder)
        disease_counter += 1
        # copy to disease folder
    else:
        # copy to normal folder
        shutil.copy(image_names[img], normal_folder)
        normal_counter += 1
    

print('end')