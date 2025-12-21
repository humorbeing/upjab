from upjab_ActGP.data.design_data.load_data_module_v08 import LoadDATA
# from efficientGP.metric.design_data.eval_module import evaluate_result
from upjab_ActGP.metric.design_data.eval_module_v03 import evaluate_result



from upjab_ActGP.train_val_test.utils import DIVIDE

from upjab_ActGP.train_val_test.one_GP_inference_module import (
    one_GP_inference,
)
from upjab_ActGP.models.gp_design.gp_model_from_config_module_02 import get_gp_model

def one_train_sciGP_model(    
    x_train,
    y_train,    
):   
    
    model = get_gp_model()
    model.fit(x_train, y_train)        
    
    return model


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
        model = one_train_sciGP_model(            
            data.x_trainval,
            data.y_trainval[:, i],
            # data.x_val,
            # data.y_val[:, i],
            # # EPOCH=EPOCH,
            # config
        )

        y_pred = one_GP_inference(
            model, x_test=x_test
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
    "FeatureAugmentation_Runs": 3,
    "x_transformer_type": "standardize",
    # "x_transformer_type": "normalize",
    "y_transformer_type": "standardize",
    # "y_transformer_type": "normalize",    
}

from omegaconf import OmegaConf
config = OmegaConf.create(config)


config.num_trainval_slice = 40

# config.num_inducing_point = 50
config.EPOCH = 500

config.exp_name = "DGP06_SciGP_Far3"
# config.exp_name = "test_run_t0005"

exp_N_trainval(
    num_trainval_slice=config.num_trainval_slice,
    config=config)


# exp_run(config=config)

print("done")
