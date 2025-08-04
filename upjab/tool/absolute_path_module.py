from upjab import ROOT
import os

# Absolute Path (AP) function
def AP(relative_path):    
    return os.path.join(ROOT, relative_path)


if __name__ == "__main__":    
    target_path = 'data/configs/test.yaml'
    from upjab.tool.absolute_path_module import AP
    target_path = AP(target_path)
    
    from omegaconf import OmegaConf
    
    conf = OmegaConf.load(target_path)
    print(conf)
    print(conf.lr)