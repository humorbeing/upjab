__version__ = "0.1.1"  # 20250417
__version__ = "0.1.2"  # 20250428
__version__ = "0.1.3"  # 20250709
__version__ = "0.1.4"  # 20250804


import os
ROOT = os.path.dirname(__file__)
# ROOT = os.path.dirname(ROOT)
# ROOT = os.path.dirname(ROOT)
ROOT = os.path.dirname(ROOT)

# Absolute Path (AP) to ROOT function
def AP(relative_path):    
    return os.path.join(ROOT, relative_path)

# from xxx import AP
# target_file = AP('xxxxx')

# target_file = 'xxxxx'
# from xxx import AP
# target_file = AP(target_file)

from upjab import tool

# from pathlib import Path
# ROOT = Path(__file__).parent.parent


# import os
# ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# from upjab.tool.timer import timer
# from upjab.tool.get_file_list import get_file_list
# from upjab.tool.only_python_files import only_python_files
# from upjab.tool.common import remove_folder

# def get_file_list(args, kwargs):
#     from upjab.tool.get_file_list import get_file_list
#     return get_file_list(args, **kwargs)


print("From upjab: do not add heavy modules in __init__.py. It will Load")
