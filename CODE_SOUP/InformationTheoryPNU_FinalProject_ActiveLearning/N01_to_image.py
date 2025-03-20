from torchvision import datasets
import os

root = os.path.dirname(__file__)
working_folder = f'{root}/data'

cifar10 = datasets.CIFAR10(f'{working_folder}/cifar10', train=True, download=True)

save_folder = f'{working_folder}/color/00_start_dataset'
os.makedirs(save_folder, exist_ok=True)


for i in range(len(cifar10)):
    img_name = f'{i:08d}.png'
    target = cifar10[i][1]
    folder_path = f'{save_folder}/{cifar10[i][1]}'
    os.makedirs(folder_path, exist_ok=True)
    cifar10[i][0].save(f'{folder_path}/{img_name}')



print('done')