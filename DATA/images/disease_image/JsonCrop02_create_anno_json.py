import json

path = f'JsonCrop_annotations.json'

with open(path, 'r') as f:
    data = json.load(f)

categories = data['categories']
images = data['images']
annotations = data['annotations']

rock_beam_normal_list = []
rock_beam_ab_list = []

images_dict = {}
for im in images:
    im_id = int(im['id'])
    images_dict[im_id] = im

ann_dict = {}

for i in data['annotations']:
    cate_id = i['category_id']
    if cate_id == 3:
        is_disease = i['diseases_exist']        
        img_id = int(i['image_id'])
        img_info = images_dict[img_id]        
        check_image_id = int(images_dict[img_id]['id'])
        if img_id == check_image_id:
            pa = images_dict[img_id]['file_name']
            pat = pa.split('/')[-1]
            ann_dict[pat] ={
                'image': img_info,
                'annotation': i
            }

with open(f'JsonCrop_dataset_annotations.json', 'w') as f:
    json.dump(ann_dict, f)
print('end') 

