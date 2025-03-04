import json

path = f'JsonCrop_annotations.json'

with open(path, 'r') as f:
    data = json.load(f)

categories = data['categories']
images = data['images']
annotations = data['annotations']


print('')