from upjab_ActGP.data.design_data.load_data_module_v08 import LoadDATA
# from efficientGP.metric.design_data.eval_module import evaluate_result
from upjab_ActGP.metric.design_data.eval_module_v03 import evaluate_result
from upjab_ActGP.models.gp_design.gpytorch_ExactGP_v02 import Get_GP_model
# from efficientGP.models.gp_design.gpytorch_DNN_ExactGP_v02 import Get_GP_model
from upjab_ActGP.train_val_test.one_gpytorch_inference_module import (
    one_gpytorch_inference,
)

from upjab_ActGP.train_val_test.utils import DIVIDE

import gpytorch




def one_train_gpytorch_model(    
    x_train,
    y_train,
    x_val,
    y_val,
    EPOCH=1000,
):
    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    x_train = torch.Tensor(x_train).to(device)
    y_train = torch.Tensor(y_train).to(device)
    # x_test = torch.Tensor(x_test).to(device)
    # y_test = torch.Tensor(y_test).to(device)
    x_val = torch.Tensor(x_val).to(device)
    y_val = torch.Tensor(y_val).to(device)

    gp_model = Get_GP_model(
        x_train=x_train, y_train=y_train
    )

    gp_model = gp_model.to(device)

    optimizer = torch.optim.Adam(
        gp_model.model.parameters(),
        lr=0.05,
    )

    import gpytorch

    mll = gpytorch.mlls.ExactMarginalLogLikelihood(
        gp_model.likelihood, gp_model.model)

    best_val_loss = float("inf")
    best_model = None

    for epoch in range(EPOCH):
        gp_model.train()
        optimizer.zero_grad()
        output = gp_model.model(x_train)
        loss = -mll(output, y_train)
        # print('Iter %d/%d - Loss: %.3f' % (epoch + 1, EPOCH, loss.item()))
        loss.backward()
        optimizer.step()

        gp_model.eval()
        # mll = gpytorch.mlls.ExactMarginalLogLikelihood(gp_model.likelihood, gp_model.model)
        with torch.no_grad():
            val_output = gp_model.model(x_val)
            val_loss = -mll(val_output, y_val)
            # print('Iter %d/%d - Val Loss: %.3f' % (epoch + 1, EPOCH, val_loss.item()))

            if val_loss.item() < best_val_loss:
                best_val_loss = val_loss.item()
                from copy import deepcopy

                best_model = deepcopy(gp_model)
                best_epoch = epoch
            print(
                f"Best Val Loss {best_val_loss:.3f}(epoch {best_epoch+1}). Current Val Loss {val_loss.item():.3f}"
            )
    # print(f'Best epoch {best_epoch+1}')
    return best_model


def experiment_run(
    num_trainval=900,
    is_random_sample=False,
    which_FeatureAugmentation=3,
    FeatureAugmentation_Runs=1,
    x_transformer_type="standardize",
    y_transformer_type="standardize",
    num_inducing_point=500,
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
    for i in range(data.y_train.shape[1]):
        gp_model = one_train_gpytorch_model(            
            data.x_trainval,
            data.y_trainval[:, i],
            data.x_val,
            data.y_val[:, i],
            EPOCH=EPOCH,
        )

        y_pred = one_gpytorch_inference(
            gp_model.model, gp_model.likelihood, x_test=x_test
        )
        y_pred_list.append(y_pred)

    import numpy as np

    y_preds = np.stack(y_pred_list, axis=-1)
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
    config
):
    num_divide = DIVIDE[0]
    num_random_rounds = DIVIDE[1]
    
    for i in range(len(num_divide)):

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

# config.num_inducing_point = 50
config.EPOCH = 100

config.exp_name = "DGP17_ExactGPv2d2_Far2"
# config.exp_name = "test_run_t0005"

exp_N_trainval(
    num_trainval_slice=config.num_trainval_slice,
    config=config)


# exp_run(config=config)

print("done")
