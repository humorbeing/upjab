


def random_name(name):
    from datetime import datetime
    import string
    import random

    now = datetime.now()
    dt_string = now.strftime("%Y-%m-%d__%H-%M-%S")

    letters = string.ascii_letters
    random_string = ''.join(random.choice(letters) for i in range(10))    

    logname = dt_string + '__' + random_string + '__' + name
    return logname


def keep_log(args, log_dir='logs'):
    import logging
    
    import os 

    signature = args.log_name
    logname = signature + '.log'
    log_path = f'{args.working_path}/{log_dir}'
    os.makedirs(log_path, exist_ok=True)

    path = os.path.join(log_path, logname)
    
    logging.basicConfig(filename=path, level=logging.INFO, filemode='w')
    logging.info("starting: " + signature)
    return logging

def log_args(logging, args):
    for key in vars(args):
        value = vars(args)[key]
        msg = f'{key}: {value}'
        logging.info(msg)
    
    logging.info('----------------------------------------')
    logging.info('----------------------------------------')


def set_seed(seed):
    import torch
    import numpy as np
    import random
    
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = True




def args_setup_and_logger(args):
    logging = keep_log(args, log_dir=args.log_save_folder)
    log_args(logging, args)

    if args.seed > 0:
        set_seed(args.seed)
    import os
    os.makedirs(f'{args.working_path}/{args.checkpoint_save_folder}', exist_ok=True)
    return logging



if __name__ == '__main__':
    from experiment_configurations.v0001.args_setup import args_setup
    args = args_setup("From_Experiment_Args_Setup")
    logging = args_setup_and_logger(args)
    logging.info(f'fake accuracy: 0.58 fake auc-roc: 0.88 fake loss: 0.42')
    logging.info("done")
    with open(args.checkpoint_save_path, 'w') as f:
        f.write('fake checkpoint')
    print('')