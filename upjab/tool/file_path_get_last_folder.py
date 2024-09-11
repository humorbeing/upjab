import os


def file_path_get_last_folder(fi, surfix='_ADDSurFix'):
    folder_path_list = fi.split('/')[:-1]
    if folder_path_list:
        pass
    else:
        folder_path_list = ['.']

    folder_path = os.path.join(*folder_path_list) + surfix
    if fi.startswith('/'):
        folder_path = f'/{folder_path}'
    
    return folder_path

if __name__ == '__main__':
    from upjab.tool.get_file_path_list import get_file_path_list
    target_folder = 'example_data/images'
    file_list = get_file_path_list(target_folder, ['py'])
    # print(file_list)
    for f_ in file_list:
        new_file_path = file_path_get_last_folder(f_, '_ALLGOOD')
        print(f_)
        print(new_file_path)
        print('--')
    print('End')
