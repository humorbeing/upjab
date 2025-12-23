# from efficientGP.metric.MAPE import MAPE
from ALGP.scikit_regression_metrics import compute_sklearn_regression_metrics

def make_logs(log_name, log_results):
    log_list = []
    for key, value in log_results.items():
        log = f"{log_name} | {key}: {value:.6f}"
        print(log)
        log_list.append(log)
    return log_list


def evaluate_result(y_pred, y_true):

    logs_list = []
    for i in range(y_true.shape[1]):
        mape = compute_sklearn_regression_metrics(y_true=y_true[:, i], y_pred=y_pred[:, i])
        # mape_list.append(mape)
        logs = make_logs(f"Output y[{i}]", mape)
        # print(logs)
        logs_list.extend(logs)    
    
    return logs_list


