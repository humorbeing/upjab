import glob


def get_file_list(target_folder, file_extends=["jpg", "JPG"]):
    """
    Retrieves a list of files with specified extensions from a target folder and its subdirectories.

    Args:
        target_folder (str): The folder to search for files.
        file_extends (list, optional): A list of file extensions to include in the search.
                                       Defaults to ['jpg', 'JPG'].

    Returns:
        list: A list of file paths matching the specified extensions.
    """

    file_list = []
    for ext_ in file_extends:
        files = glob.glob(target_folder + f"/**/*.{ext_}", recursive=True)
        file_list.extend(files)

    return file_list


if __name__ == "__main__":
    target_folder = "data/text"
    file_list = get_file_list(target_folder, file_extends=["txt"])
    for f_ in file_list:
        print(f_)
    print("End")
