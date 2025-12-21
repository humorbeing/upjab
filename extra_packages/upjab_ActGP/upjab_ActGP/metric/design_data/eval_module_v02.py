# from efficientGP.metric.MAPE import MAPE
from upjab_ActGP.metric.scikit_regression_metrics import compute_sklearn_regression_metrics
import numpy as np
from upjab_ActGP.metric.design_data.calculate_efficiency_module import calculate_efficiency


def make_logs(log_name, log_results):
    log_list = []
    for key, value in log_results.items():
        log = f"{log_name} | {key}: {value:.6f}"
        print(log)
        log_list.append(log)
    return log_list


def evaluate_result(y_pred, y_true, x_test):
    names = ['Pt_in [Pa]', 'Pt_out [Pa]', 'del_Pt [Pa]', 'Torque [N m]', 'Efficiency [%]']
    
    # mape_list = []
    logs_list = []
    for i in range(y_true.shape[1]):
        mape = compute_sklearn_regression_metrics(y_pred[:, i], y_true[:, i])
        # mape_list.append(mape)
        logs = make_logs(names[i], mape)
        # print(logs)
        logs_list.extend(logs)
    
    
    e1, e2, e3 = calculate_efficiency(x_test, y_pred)

    mape = compute_sklearn_regression_metrics(e1, y_true[:,4])
    logs = make_logs('Efficiency (Pt_out - Pt_in)', mape)
    logs_list.extend(logs)
    
    mape = compute_sklearn_regression_metrics(e2, y_true[:,4])
    logs = make_logs('Efficiency (Pt_out)', mape)
    logs_list.extend(logs)

    mape = compute_sklearn_regression_metrics(e3, y_true[:,4])
    logs = make_logs('Efficiency (del_Pt)', mape)
    logs_list.extend(logs)
    
    return logs_list


