__version__ = "0.0.1"



import os
ROOT = os.path.dirname(__file__)
# ROOT = os.path.dirname(ROOT)
# ROOT = os.path.dirname(ROOT)
ROOT = os.path.dirname(ROOT)

# Absolute Path (AP) to ROOT function
def AP(relative_path):    
    return os.path.join(ROOT, relative_path)