
def log_results(
    exp_setup,
    results_list="",
    log_folder=None,
):

    from loguru import logger
    if log_folder is None:
        from upjab_ActGP import AP
        log_folder = AP('logs_W_random')
        
        
    exp_name = exp_setup.exp_name
    num_trainval = exp_setup.num_trainval
    random_round_num = exp_setup.random_round_num
    log_path = f"{log_folder}/{exp_name}/{num_trainval}/{random_round_num}/" + "{time}_" + f"{exp_name}.log"
    
        

    handler_id = logger.add(log_path)

    for k, v in exp_setup.items():

        logger.info(f"{k}: {v}")

    for l in results_list:
        logger.info(l)

    logger.remove(handler_id)
