import numpy as np
from ALGP.get_one_gp_model_module import get_gp_model
from ALGP.utils import pop_selected_list

def ALGP_training_set(
    x_train: np.ndarray,
    y_train: np.ndarray,
    ActiveLearning_Target_Index: int = 0,
    Budget_Training_Sample: int = 50,
    NUM_INIT_SAMPLE: int = 10,
    TOP_K: int = 1,
    gp_config_path = None,
):
    assert 1 <= Budget_Training_Sample <= len(x_train), "Budget_Training_Sample must be between 1 and the size of the training set"
    assert 1 <= NUM_INIT_SAMPLE <= Budget_Training_Sample, "NUM_INIT_SAMPLE must be between 1 and Budget_Training_Sample"
    assert 1 <= TOP_K <= Budget_Training_Sample, "TOP_K must be between 1 and Budget_Training_Sample"
    assert x_train.ndim == 2, "x_train must be a 2D array"
    assert y_train.ndim == 2, "y_train must be a 2D array"
    len_x_feature = x_train.shape[-1]
    len_y_feature = y_train.shape[-1]

    training_dataset = np.concatenate((x_train, y_train), axis=1)
    selected_index_list = np.random.choice(len(training_dataset), NUM_INIT_SAMPLE, replace=False)

    x_training = None
    y_training = None

    remaining = training_dataset    

    while True:
        result_list = pop_selected_list(
            original_array=remaining,
            index_list=selected_index_list
        )

        selected = result_list['selected']
        remaining = result_list['remaining']


        x_trainingset_selected = selected[:,:len_x_feature]
        y_trainingset_selected = selected[:, -len_y_feature:]

        if x_training is None:
            x_training = x_trainingset_selected
            y_training = y_trainingset_selected
        else:
            x_training = np.concatenate((x_training, x_trainingset_selected), axis=0)
            y_training = np.concatenate((y_training, y_trainingset_selected), axis=0)
        
        if len(x_training) >= Budget_Training_Sample:
            return {
                'x_training': x_training,
                'y_training': y_training
            }
        
        else:
            x_evaluate = remaining[:,:len_x_feature]

            model = get_gp_model(gp_config_path)
            model.fit(x_training, y_training[:, ActiveLearning_Target_Index])
    
            
            y_mean, y_std = model.predict(x_evaluate, return_std=True)  
            selected_index_list = np.argpartition(y_std, -TOP_K)[-TOP_K:]
