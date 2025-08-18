import os




# Dynamic Path (DP) function
def DP(relative_path, current_path, num_subdirs=1):
    current_file_path = os.path.dirname(current_path)    
    relative_folder = os.path.join(current_file_path, *['..'] * num_subdirs)
    return os.path.join(relative_folder, relative_path)


if __name__ == "__main__":    
    target_path = 'data/configs/test.yaml'
    from upjab.tool.dynamic_path_module import DP
    target_path = DP(target_path, __file__, 2)
    
    from omegaconf import OmegaConf
    
    conf = OmegaConf.load(target_path)
    print(conf)
    print(conf.lr)
