from upjab_ActGP.data.design_data.load_data_module_v08 import LoadDATA
# from efficientGP.metric.design_data.eval_module import evaluate_result
from upjab_ActGP.metric.design_data.eval_module_v03 import evaluate_result
# from efficientGP.models.gp_design.gpytorch_ExactGP_v01 import Get_GP_model
# from efficientGP.models.gp_design.gpytorch_DNN_ExactGP_v01 import Get_GP_model
from upjab_ActGP.train_val_test.one_gpytorch_inference_module import (
    one_gpytorch_inference,
)

from upjab_ActGP.train_val_test.utils import DIVIDE

import gpytorch


from gpytorch.mlls import DeepApproximateMLL
from gpytorch.mlls import VariationalELBO

from upjab_ActGP.models.gp_design.gpytorch_DGP_v01 import DeepGP_model
from upjab_ActGP.train_val_test.one_DGP_gpytorch_inference_module import (
    one_DGP_gpytorch_inference,
)

def train_DNN_model(    
    x_train,
    y_train,
    x_val,
    y_val,
    EPOCH=100,
    nn_structure = (16, 32, 64, 64, 32, 16),
    lr=1e-3,
    dropout=0.1,
    validation_interval=5
):   
    
    from upjab_ActGP.models.gp_design.deep_NN_v01 import MLPRegressor
    import torch
    in_dim = x_train.shape[1]
    if y_train.ndim == 1:
        out_dim = 1
    else:
        out_dim = y_train.shape[1]
    
    model = MLPRegressor(in_dim=in_dim, hidden=nn_structure, out_dim=out_dim, dropout=dropout)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")        
    model.to(device)
    # Loss & Optimizer
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    # (optional) scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)
    
    train_data = {
        'x': torch.tensor(x_train, dtype=torch.float32),
        'y': torch.tensor(y_train, dtype=torch.float32)
    }

    val_data = {
        'x': torch.tensor(x_val, dtype=torch.float32),
        'y': torch.tensor(y_val, dtype=torch.float32)
    }
    best_val_loss = float('inf')
    from upjab_ActGP.train_val_test.deeplearning_v01 import train_one_step
    from copy import deepcopy
    for epoch in range(1, EPOCH+1):
        
        train_loss = train_one_step(train_data, model, criterion, optimizer)    
        val_loss = train_one_step(val_data, model, criterion, optimizer=None)
        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model = deepcopy(model)

        if epoch % validation_interval == 0 or epoch == 1:
            
            print(f"Epoch {epoch}/{EPOCH}, Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
            
    return best_model



def experiment_run(
    num_trainval=900,
    is_random_sample=False,
    which_FeatureAugmentation=3,
    FeatureAugmentation_Runs=1,
    x_transformer_type="standardize",
    y_transformer_type="standardize",
    num_inducing_point=500,
    nn_structure = (200,),
    EPOCH=2000,
    DOE_csv_path="data/Pump_data/DOE_data.csv",
    USE_ABSOLUTE_ROOT_PATH=True,
    **kwargs,
):
    if kwargs:
        print(f"Unexpected arguments: {kwargs}")

    import torch

    data = LoadDATA(
        num_trainval=num_trainval,
        is_random_sample=is_random_sample,
        which_FeatureAugmentation=which_FeatureAugmentation,
        FeatureAugmentation_Runs=FeatureAugmentation_Runs,
        x_transformer=x_transformer_type,
        y_transformer=y_transformer_type,
        DOE_csv_path=DOE_csv_path,
        USE_ABSOLUTE_ROOT_PATH=USE_ABSOLUTE_ROOT_PATH,
    )

    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    y_pred_list = []
    

    x_test = data.x_test
    model = train_DNN_model(            
        data.x_trainval,
        data.y_trainval,
        data.x_val,
        data.y_val,
        EPOCH=EPOCH,
        nn_structure = nn_structure,
        # config
    )
    import torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")        
    with torch.no_grad():
        model.eval()
        
        x_tensor = torch.tensor(x_test, dtype=torch.float32).to(device)
        y_preds = model(x_tensor).cpu().numpy()
    import numpy as np

    # y_preds = np.stack(y_pred_list, axis=-1)
    y_pred = data.reverse_y(y_preds)

    x_test_original = data.reverse_x(x_test)
    y_true = data.original_y_test

    results_list = evaluate_result(y_pred, y_true, x_test_original)

    return results_list


def log_results(
    exp_setup,
    results_list="",
    log_folder=None,
):

    from loguru import logger
    if log_folder is None:
        from upjab_ActGP import AP
        log_folder = AP('logs_W_random')
        
        
    exp_name = exp_setup.exp_name
    num_trainval = exp_setup.num_trainval
    random_round_num = exp_setup.random_round_num
    log_path = f"{log_folder}/{exp_name}/{num_trainval}/{random_round_num}/" + "{time}_" + f"{exp_name}.log"
    
        

    handler_id = logger.add(log_path)

    for k, v in exp_setup.items():

        logger.info(f"{k}: {v}")

    for l in results_list:
        logger.info(l)

    logger.remove(handler_id)


def exp_random_run(config):
    results_list = experiment_run(**config)
    log_results(exp_setup=config, results_list=results_list)




def exp_N_trainval(
    num_trainval_slice,
    config,
    start_from=0
):
    num_divide = DIVIDE[0]
    num_random_rounds = DIVIDE[1]
    
    for i in range(start_from, len(num_divide)):

        random_rounds = num_random_rounds[i]
        n_trainval = num_divide[i]
        # config.num_trainval = n_trainval
        config.num_trainval = int(n_trainval)

        for r in range(random_rounds):
            config.random_round_num = r + 1
            exp_random_run(config=config)


config = {
    "num_trainval": 100,
    "is_random_sample": True,
    "which_FeatureAugmentation": 3,
    "FeatureAugmentation_Runs": 2,
    "x_transformer_type": "standardize",
    # "x_transformer_type": "normalize",
    "y_transformer_type": "standardize",
    # "y_transformer_type": "normalize",    
}

from omegaconf import OmegaConf
config = OmegaConf.create(config)

config.num_trainval_slice = 40
config.num_inducing_point=900

# config.num_inducing_point = 50
config.EPOCH = 500
config.nn_structure = (64, 64,128,128,64,64)
config.exp_name = "DGP24_DNN_6layer_Far2"
# config.exp_name = "test_run_t0005"

exp_N_trainval(
    num_trainval_slice=config.num_trainval_slice,
    config=config,
    start_from=0)


# exp_run(config=config)

print("done")
