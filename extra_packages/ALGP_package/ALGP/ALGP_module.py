from ALGP.gp_model_module import GPModel
from ALGP.ALGP_loop import ALGP_training_set
from ALGP import AP
import numpy as np


def ALGP(
    x_train: np.ndarray,
    y_train: np.ndarray,
    index_selection_target = 0,
    Budget_Training_Sample = 50,
    NUM_INIT_SAMPLE=10,
    TOP_K = 1,
    gp_config_path = AP('configs/gp-l_bfgs_v01.yaml'),
):
    
    train_dataset = ALGP_training_set(
        x_train=x_train,
        y_train=y_train,
        index_selection_target = index_selection_target,
        Budget_Training_Sample = Budget_Training_Sample,
        NUM_INIT_SAMPLE=NUM_INIT_SAMPLE,
        TOP_K = TOP_K,
        gp_config_path = gp_config_path,
    )

    x_training = train_dataset['x_training']
    y_training = train_dataset['y_training']

    model = GPModel(
        x_training=x_training,
        y_training=y_training,
        gp_config_path=gp_config_path,
    )
    return model