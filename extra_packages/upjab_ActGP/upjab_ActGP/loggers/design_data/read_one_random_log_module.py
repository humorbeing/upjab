import glob
import numpy as np

def get_one_sample_metric_value(
    target_folder,
    output_name = 'Efficiency (Pt_out - Pt_in)',
    metric_name = 'mean_absolute_percentage_error',
):
    
    file_list = glob.glob(target_folder + "/**/*.log", recursive=True)
    values = []
    for file_path in file_list:
        with open(file_path, "r") as file:
            lines = file.readlines()
        check_string = f'{output_name} | {metric_name}'
        for l in lines:
            if check_string in l:
                
                # print(l)
                value = float(l.strip().split(' ')[-1])
        values.append(value)
        

    
    values = np.array(values)
    
    mean_value = np.mean(values)
    std_value = np.std(values)
    median_value = np.median(values)
    return {
        'values': values,
        'mean': mean_value,
        'std': std_value,
        'median': median_value,
    }


if __name__ == '__main__':
    target_folder = 'logs_W_random/test_run_t0004/10'
    result = get_one_sample_metric_value(
        target_folder,
        output_name = 'Efficiency (Pt_out - Pt_in)',
        metric_name = 'mean_absolute_percentage_error',
    )
    print(result)
    print("break")
