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
    ax,
    target_folder = "logs_W_random/test_run_t0004",
    output_name = "Efficiency (Pt_out - Pt_in)",
    metric_name = "mean_absolute_percentage_error",
    exp_name = "Deep NeuralNet",
    color = "#434348",
    linestyle = '-.',
    y_scale = 1.0,
    y_sigma=0.8,
    y_std_sigma=1.2
):
    random_rounds_logs = get_random_rounds_logs(
        target_folder,
        output_name=output_name,
        metric_name=metric_name,
    )


    draw_plot_random(
        ax=ax,
        log_dict=random_rounds_logs,
        exp_name=exp_name,
        color=color,
        linestyle=linestyle,
        scale_y=y_scale,
        y_sigma=y_sigma,
        y_std_sigma=y_std_sigma
    )






plt.rcParams.update({
    'font.family': 'Courier New',  # monospace font
    'font.size': 20,
    'axes.titlesize': 20,
    'axes.labelsize': 20,
    'xtick.labelsize': 20,
    'ytick.labelsize': 20,
    'legend.fontsize': 20,
    'figure.titlesize': 20
}) 
# fig, ax = plt.subplots(figsize=(10, 10))
fig, ax = plt.subplots(figsize=(7.5,7.5))

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


"#12436D"
"#005CAB"
"#28A197"
"#F46A25"
"#E31B23"
"#3D3D3D"
"#FFC325"

"#125A56"
"#00767B"
"#238F9D"
"#FFB954"
"#FD9A44"
"#F57634"
"#E94C1F"
"#D11807"

"#c8c8c8"
"#f0c571"
"#59a89c"
"#0b81a2"
"#e25759"
"#9d2c00"
"#7e4794"
"#36b700"


'-'
'--'
'-.'
':'

output_name = "Efficiency (Pt_out - Pt_in)"
# output_name = 'Torque [N m]'
metric_name = "mean_absolute_percentage_error"
# metric_name = "rmsle"


plot_random_logs(
    ax=ax,
    target_folder = "logs_demo/DGP05_SciGP_Far2_v2",
    output_name = 'Efficiency [%]',    
    metric_name = metric_name,
    exp_name = "SciGP Predict",
    color = "#36b700",
    linestyle = '--',
    y_scale = 100.0,
    y_sigma=0.5,
    y_std_sigma=0.4
)




plot_random_logs(
    ax=ax,
    target_folder = "logs_demo/DGP05_SciGP_Far2_v2",
    output_name = output_name,    
    metric_name = metric_name,
    exp_name = "SciGP Far2",
    color = "#D11807",
    linestyle = '--',
    y_scale = 100.0,
    y_sigma=0.5,
    y_std_sigma=0.4
)



ax.set_xlabel('Number of Training Data Used')
ax.set_ylabel('MAPE of Efficiency(%)')
# ax.set_aspect('equal', adjustable='datalim') # Lock the square shape

# Major grid:
ax.grid(True, which='major', linestyle='-', linewidth=0.75, alpha=0.25)

# Minor ticks and grid:
ax.minorticks_on()
ax.grid(True, which='minor', linestyle='-', linewidth=0.25, alpha=0.15)

ax.set_axisbelow(True) # <-- Ensure grid is below data
# ax.set_ylim(0.5, 1)
ax.set_ylim(0.5, 70)
# ax.set_xlim(0, 150)
# ax.set_xscale('log', base=2)
ax.set_yscale('log')
ax.legend()

# save_folder = AP('assets/design_data/plots/temp')
save_folder = AP('assets/design_data/plots/paper_figures_v2_reduceTo150')

import os
os.makedirs(save_folder, exist_ok=True)
surfix = ""
# surfix = "_ALL6"
save_name=f"PaperFig01_Eff_Allshow{surfix}"

plt.savefig(f"{save_folder}/{save_name}.png", dpi=100)
plt.savefig(f"{save_folder}/{save_name}.pdf")
plt.savefig(f"{save_folder}/{save_name}.svg")
# plt.show(block=True)
plt.close()


print("done")