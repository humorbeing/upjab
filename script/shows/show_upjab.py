import upjab



def show_upjab1():
    with upjab.timer:

        target_folder = 'data/text'

        file_list = upjab.get_file_list(target_folder, file_extends=['txt'])
        for f_ in file_list:
            print(f_)
        

        target_folder='extra_packages/upjab_FirstPackage'

        upjab.only_python_files(target_folder=target_folder)


def clean_upjab1():
    target_folder = 'extra_packages/upjab_FirstPackage_OnlyPythonFile'
    upjab.remove_folder(target_folder)
    print('-------- Cleaned --------')


if __name__ == '__main__':
    show_upjab1()
    clean_upjab1()