import torch


def one_GP_inference(
    model,    
    x_test,    
):    
    y_pred, y_std = model.predict(x_test, return_std=True)
    
    return y_pred
