
# import upjab.tool  # change to upjab.__init__.py
import upjab # works. looks like loading cannot be avoided


def show_upjab1():
    with upjab.tool.timer:

        target_folder = "data/text"

        file_list = upjab.tool.get_file_list(target_folder, file_extends=["txt"])
        for f_ in file_list:
            print(f_)

        target_folder = "extra_packages/upjab_FirstPackage"

        upjab.tool.only_python_files(target_folder=target_folder)
        file_path = "data/configs/learn_yaml.yaml"
        data = upjab.tool.yaml(file_path)
        print(data)
        print("Data loaded successfully.")


def clean_upjab1():
    target_folder = "extra_packages/upjab_FirstPackage_OnlyPythonFile"
    upjab.tool.remove_folder(target_folder)
    print("-------- Cleaned --------")


if __name__ == "__main__":
    show_upjab1()
    clean_upjab1()
