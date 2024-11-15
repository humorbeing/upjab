import argparse
import os
from upjab.tool.args_setup_and_logger import random_name


def args_setup(
    exp_name='NeedExperimentName',
    working_path=None,
    checkpoint_save_folder='checkpoints',
    log_save_folder='logs',):    

    args = argparse.ArgumentParser()




    # training environment setup
    if working_path is None:
        working_path = os.path.dirname(os.path.abspath(__file__))
    
    args.working_path = working_path    
    args.checkpoint_save_folder = checkpoint_save_folder
    args.log_save_folder = log_save_folder
    args.log_name = random_name(exp_name)
    args.seed = 0  # some problems with random seed
    args.checkpoint_save_path = f'{working_path}/{checkpoint_save_folder}/{args.log_name}_best_model_seed_{args.seed}.pkl'

    
    
    
    # training hyper parameter setup
    args.num_steps = 1000 + 1
    # args.lr = [0.0001] * args.num_steps
    args.lr = 0.001
    args.batch_size = 372  # half of 
    args.evaluate_freq = 20
    
    args.load_pretrained = False
    args.model_ckpt = '.'
    



    # model hyper parameter setup
    args.num_input_features = 1024





    
    return args


if __name__ == '__main__':
    args = args_setup("ExperimentOf_HelloWorld")
    print(args.__dict__)
