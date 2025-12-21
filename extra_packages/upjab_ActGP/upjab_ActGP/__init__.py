# __version__ = "0.0.1"  # 20250804
# __version__ = "0.0.2"  # 20250804
# __version__ = "0.0.3"  # 20251111
# __version__ = "0.1.0"  # 20251221
__version__ = "0.0.1"  # 20251221

import os
ROOT = os.path.dirname(__file__)
ROOT = os.path.dirname(ROOT)

# Absolute ROOT Path (AP) function
def AP(relative_path):
    return os.path.join(ROOT, relative_path)


# target_file = 'xxxxx'
# from xxx import ARP
# target_file = ARP(target_file)