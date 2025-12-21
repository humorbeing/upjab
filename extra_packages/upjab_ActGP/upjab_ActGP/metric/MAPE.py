import numpy as np


# def MAPE(y_true, y_pred):
#     error_p = np.mean(np.abs((y_true - y_pred)/y_true)) * 100
    
#     return error_p

def MAPE(y_pred, y_true):
    error_p = np.mean(np.abs((y_true - y_pred)/y_true)) * 100
    
    return error_p



    
