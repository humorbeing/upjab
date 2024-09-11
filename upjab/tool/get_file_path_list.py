import glob

def get_file_path_list(target_folder, file_extends=['jpg', 'JPG']):
    file_list = []
    for ext_ in file_extends:
        files = glob.glob(target_folder + f'/**/*.{ext_}', recursive=True)        
        file_list.extend(files)
    
    return file_list
    

if __name__ == '__main__':
    target_folder = '../../example_data/images'
    file_list = get_file_path_list(target_folder, file_extends=['jpg', 'JPG', 'json'])
    for f_ in file_list:
        print(f_)
    print('End')