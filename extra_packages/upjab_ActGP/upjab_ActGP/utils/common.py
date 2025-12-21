import os

# Dynamic Path (DP) Module
def DP(relative_path, current_path, num_subdirs=1):
    current_file_path = os.path.dirname(current_path)    
    relative_folder = os.path.join(current_file_path, *['..'] * num_subdirs)
    return os.path.join(relative_folder, relative_path)


def params_count(model):         
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Number of trainable parameters: {trainable_params}")

    print("Trainable parameters in the model:")
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"Name: {name}, Shape: {param.shape}")

if __name__ == "__main__":
    from upjab_ActGP.data.load_design_module import load_design_csv


    target_file = 'data/Pump_data/design_variable.csv'
    from upjab_ActGP.utils.common import DP
    target_file = DP(target_file, __file__, 2)

    
    design_keys, design_variable = load_design_csv(target_file)
    print(design_keys)
    print(design_variable[0])
