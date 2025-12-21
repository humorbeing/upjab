from upjab_ActGP.data.design_data.load_data_module_v08 import LoadDATA
from upjab_ActGP.metric.design_data.eval_module_v03 import evaluate_result

from upjab_ActGP.train_val_test.one_GP_inference_module import (
    one_GP_inference,
)
import numpy as np
from upjab_ActGP.models.gp_design.gp_model_from_config_module_02 import get_gp_model

from upjab_ActGP.loggers.design_data.write_log_randomsample_module import log_results


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

config.random_round_num = 1


config.exp_name = "DGP21_ActGPv2_Far2"
# config.exp_name = "test_run_t0006"



def one_train_sciGP_model(    
    x_train,
    y_train,    
):   
    
    model = get_gp_model()
    model.fit(x_train, y_train)        
    
    return model


def experiment_run(
    x_trainval,
    y_trainval,
    data
):    
    y_pred_list = []    

    x_test = data.x_test
    for i in range(data.y_train.shape[1]):
        model = one_train_sciGP_model(            
            x_trainval,
            y_trainval[:, i],            
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

    config.num_trainval = len(x_trainval)

    log_results(exp_setup=config, results_list=results_list)
    return results_list


def pop_selected_list(
    original_array,
    index_list      
):    
    selected_array = original_array[index_list]
    
    mask = np.ones(len(original_array), dtype=bool)
    
    mask[index_list] = False
    remaining_array = original_array[mask]

    return {
        'selected': selected_array,
        'remaining': remaining_array
    }






NUM_INIT_SAMPLE = 10


data = LoadDATA(
    num_trainval=900,
    is_random_sample=False,
    which_FeatureAugmentation=config.which_FeatureAugmentation,
    FeatureAugmentation_Runs=config.FeatureAugmentation_Runs,
    x_transformer='standardize',
    y_transformer='standardize',
)

x_train = data.x_trainval
y_train = data.y_trainval

len_x_feature = x_train.shape[-1]
len_y_feature = y_train.shape[-1]

training_set = np.concatenate((x_train, y_train), axis=1)

index_list = np.random.choice(len(training_set), NUM_INIT_SAMPLE, replace=False)

x_training = None
y_training = None

remaining = training_set

from upjab_ActGP.train_val_test.utils import DIVIDE_activeLearning

num_divide_list = DIVIDE_activeLearning[0]
num_divide_topK_list = DIVIDE_activeLearning[1]


TOP_K = 1

while True:
    result_list = pop_selected_list(
        original_array=remaining,
        index_list=index_list
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

    x_len = len(x_training)
    print(f'X train set length: {x_len}')
    # ind = np.where(num_divide_list==x_len)
    # TOP_K = num_divide_topK_list[ind].item()
    if x_len in num_divide_list:
        experiment_run(
            x_trainval=x_training,
            y_trainval=y_training,
            data=data
        )

    x_evaluate = remaining[:,:len_x_feature]

    model = get_gp_model()
    model.fit(x_training, y_training[:, 2])        
    
    
    
    if x_len == 200:
        break
    else:
        # index_list = np.random.choice(len(x_evaluate), TOP_K, replace=False)

        _, y_std = model.predict(x_evaluate, return_std=True)        

        index_list = np.argpartition(y_std, -TOP_K)[-TOP_K:]

