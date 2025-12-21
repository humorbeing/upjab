import glob


def read_log(file_path):

    with open(file_path, "r") as file:
        lines = file.readlines()

    break_string = " - "
    num_MAPE_line = 8
    exp_setup_lines = lines[:-num_MAPE_line]
    MAPE_lines = lines[-num_MAPE_line:]

    EXP_SETUP_log = {}
    for l in exp_setup_lines:
        temp11 = l.strip()
        start_loc = temp11.find(" - ") + 3
        temp12 = temp11[start_loc:]
        sep_loc = temp12.find(":")
        key = temp12[:sep_loc]
        value = temp12[sep_loc + 2 :]
        EXP_SETUP_log[key] = value

    MAPE_log = {}

    for l in MAPE_lines:
        temp11 = l.strip()
        start_loc = temp11.find(break_string) + 3
        end_loc = temp11.find("MAPE:") - 1
        mape_name = temp11[start_loc:end_loc]
        temp21 = temp11[end_loc + 7 : -1]
        temp22 = float(temp21)
        MAPE_log[mape_name] = temp22

    return {"EXP_SETUP_log": EXP_SETUP_log, "MAPE_log": MAPE_log}


def get_logs_from_folder(target_folder):    

    # target_folder = '.'
    file_list = glob.glob(target_folder + "/**/*.log", recursive=True)

    logs = []
    for file_path in file_list:
        log = read_log(file_path)
        logs.append(log)
    return logs


if __name__ == "__main__":
    from upjab_ActGP import AP
    logs = get_logs_from_folder(AP("logs/N_trainval_BY_N_inducing_points"))
    print(logs)
    print("break")