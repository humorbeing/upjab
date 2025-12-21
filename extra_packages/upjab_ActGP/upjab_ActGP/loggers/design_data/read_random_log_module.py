from upjab_ActGP.loggers.design_data.read_one_random_log_module import (
    get_one_sample_metric_value,
)

import os
from upjab_ActGP import AP

def get_random_rounds_logs(
    target_folder,
    output_name="Efficiency (Pt_out - Pt_in)",
    metric_name="mean_absolute_percentage_error",
):
    target_folder = AP(target_folder)
    subfolder_list = os.listdir(target_folder)
    logs = {}
    for subfolder in subfolder_list:
        target_subfolder = f"{target_folder}/{subfolder}"

        result = get_one_sample_metric_value(
            target_subfolder,
            output_name=output_name,
            metric_name=metric_name,
        )
        logs[subfolder] = result
        # print(f"Subfolder: {subfolder}, Mean MAPE: {result['mean']}, Std MAPE: {result['std']}")

    return logs


if __name__ == "__main__":
    output_name_list = [
        'Pt_in [Pa]',
        'Pt_out [Pa]',
        
        'Torque [N m]',
        'Efficiency [%]',
        'Efficiency (Pt_out - Pt_in)',
    ]



    metric_name_list = [
        'explained_variance_score',
        'r2_score',
        'd2_absolute_error_score',
        'd2_pinball_score',
        'd2_tweedie_score',
        'mean_absolute_error',
        'median_absolute_error',
        'mean_squared_error',
        'rmse',
        'mean_squared_log_error',
        'rmsle',
        'mean_absolute_percentage_error',
        'max_error',
        'mean_pinball_loss(alpha=0.5)',
        'mean_poisson_deviance',
        'mean_gamma_deviance',
        'mean_tweedie_deviance(power=0.0)',
    ]

    target_folder = "logs_W_random/test_run_t0004"
    for output_name in output_name_list:
        for metric_name in metric_name_list:
            print(f"Output: {output_name}, Metric: {metric_name}")
            
            logs = get_random_rounds_logs(
                target_folder,
                output_name=output_name,
                metric_name=metric_name,
            )

            print(logs)
            print("---------------")
    
    
    print("break")
