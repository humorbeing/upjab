from upjab_ActGP.loggers.design_data.read_log_module import get_logs_from_folder


def get_logs_array(
    logs_folder_path,
    interested_keys = [
        'num_trainval',
        # 'num_inducing_point',
    ],
):    
    logs = get_logs_from_folder(logs_folder_path)

    value_list = []

    exp_name = logs[0]['EXP_SETUP_log']['exp_name']

    for log in logs:
        # print(log)
        one_value_list = []
        MAPE_log = log['MAPE_log']
        for key, value in MAPE_log.items():
            # print(f"{key}: {value}")
            one_value_list.append(value)
        SETUP_log = log['EXP_SETUP_log']
        for key, value in SETUP_log.items():
            # print(f"{key}: {value}")
            pass
        for key in interested_keys:
            one_value_list.append(float(SETUP_log[key]))
        # print('breakpoint')
        value_list.append(one_value_list)

    import numpy as np

    log_array = np.array(value_list)

    axis_name_list = [
        'Pt_in [Pa]',
        'Pt_out [Pa]',
        'del_Pt [Pa]',
        'Torque [N m]',
        'Efficiency [%]',
        'Efficiency (Pt_out - Pt_in)',
        'Efficiency (Pt_out)',
        'Efficiency (del_Pt)',
        'num_trainval',
    ]
    for interested_key in interested_keys:
        axis_name_list.append(interested_key)
    return log_array, axis_name_list, exp_name