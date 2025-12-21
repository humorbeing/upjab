from upjab_ActGP.metric.MAPE import MAPE
import numpy as np
from upjab_ActGP.metric.design_data.calculate_efficiency_module import calculate_efficiency

def evaluate_result(y_pred, y_true, x_test):
    names = ['Pt_in [Pa]', 'Pt_out [Pa]', 'del_Pt [Pa]', 'Torque [N m]', 'Efficiency [%]']
    
    mape_list = []
    logs_list = []
    for i in range(y_true.shape[1]):
        mape = MAPE(y_pred[:, i], y_true[:, i])
        mape_list.append(mape)
        logs = f'{names[i]} MAPE: {mape:.4f}%'
        print(logs)
        logs_list.append(logs)    
    
    e1, e2, e3 = calculate_efficiency(x_test, y_pred)
    logs = f'Efficiency (Pt_out - Pt_in) MAPE: {MAPE(e1, y_true[:,4]):.4f}%'
    print(logs)
    logs_list.append(logs)

    logs = f'Efficiency (Pt_out) MAPE: {MAPE(e2, y_true[:,4]):.2f}%'
    print(logs)
    logs_list.append(logs)

    logs = f'Efficiency (del_Pt) MAPE: {MAPE(e3, y_true[:,4]):.2f}%'
    print(logs)
    logs_list.append(logs)
    return logs_list


