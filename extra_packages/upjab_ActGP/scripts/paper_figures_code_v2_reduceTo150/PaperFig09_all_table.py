from upjab_ActGP import AP

from matplotlib import pyplot as plt
from scipy.ndimage import gaussian_filter1d

import numpy as np

from upjab_ActGP.loggers.design_data.read_random_log_module import (
    get_random_rounds_logs,
)


def draw_plot_random(
    ax,
    log_dict,
    exp_name,
    color=None,
    linestyle='-',
    scale_y=1.0,
    y_sigma=0.8,
    y_std_sigma=1.2
):

    logs = []
    for si in log_dict:
        temp11 = [
            float(si),
            log_dict[si]['mean'],
            log_dict[si]['std']
        ]
        logs.append(temp11)
    log_array =  np.array(logs)
    log_array = log_array[log_array[:, 0].argsort()]
    data_x = log_array[:, 0].reshape(-1,1)
    data_y = log_array[:, 1].reshape(-1,1)
    data_y_std = log_array[:, 2].reshape(-1,1)    
    
    data_y_smoothed = gaussian_filter1d(data_y.flatten(), sigma=y_sigma)
    
    data_y_std_smoothed = gaussian_filter1d(data_y_std.flatten(), sigma=y_std_sigma)    
    
    data_y_smoothed = data_y_smoothed * scale_y
    data_y_std_smoothed = data_y_std_smoothed * scale_y

    ax.fill_between(data_x.flatten(), (data_y_smoothed - data_y_std_smoothed).flatten(), (data_y_smoothed + data_y_std_smoothed).flatten(), color=color, alpha=0.3)
    ax.plot(data_x, data_y_smoothed, 
            label=exp_name,
            linewidth = 2.6,
            color = color,
            linestyle = linestyle,                
            zorder = 2)
    

def plot_random_logs(    
    target_folder = "logs_W_random/test_run_t0004",
    output_name = "Efficiency (Pt_out - Pt_in)",    
    # exp_name = "Deep NeuralNet",    
):
    M = get_random_rounds_logs(
        target_folder,
        output_name=output_name,
        metric_name="mean_absolute_percentage_error",
    )

    R = get_random_rounds_logs(
        target_folder,
        output_name=output_name,
        metric_name='r2_score',
    )
    key_points = [
        '50',
        '100',
    ]
    row = [target_folder,]

    
    for kp in key_points:
        m1 = M[kp]['mean']
        r1 = R[kp]['mean']
        # m1 = m1 * 100.0
        m1 = round(m1, 4)
        r1 = round(r1, 4)
        row.append(m1)
        row.append(r1)
    # row[exp_name] = temp
    return row
    # print('break')


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



output_name = "Efficiency (Pt_out - Pt_in)"
# output_name = 'Torque [N m]'


target_folder_list = [    
    "logs_demo/DGP05_SciGP_Far2_v2",    
]

logs = []

for target_folder in target_folder_list:
    temp = plot_random_logs(    
        target_folder = target_folder,
        output_name = output_name,    
    )
    logs.append(temp)



import csv
import os
os.makedirs(AP('assets/design_data/plots/paper_figures_v2_reduceTo150'), exist_ok=True)
with open(AP('assets/design_data/plots/paper_figures_v2_reduceTo150/table.csv'), 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerows(logs)

print("done")