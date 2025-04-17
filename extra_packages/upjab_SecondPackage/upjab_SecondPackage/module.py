import os

def hello():
    file_path = os.path.abspath(__file__)
    file_name = os.path.basename(file_path)

    print(f"The absolute path of the file is: {file_path}")
    print(f"The name of the file is: {file_name}")

# Call the function
# hello()