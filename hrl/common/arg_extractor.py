import argparse
from pdb import set_trace

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def get_args():
    """ 
    Returns a namedtuple with arguments extracted from the command line.
    No default values are specified here to avoid double specification in
    run_experiment function. At the end no matter how the code gets executed
    run_experiment will always be run.

    :return: A namedtuple with arguments
    """
    parser = argparse.ArgumentParser(
        description='This will parse the argument and pass them as parameters to th emian experiment function')

    parser.add_argument('--not_save', action='store_true', 
            help="True means not saving the experiment")
    parser.add_argument('--folder', type=str, 
            help="The folder where the experiment will be save.Defautl is 'experiments'")
    parser.add_argument('--save_interval', type=int,
            help="The model will be saved every number of steps specified here, default: 0")
    parser.add_argument('--train_steps', type=int,
            help="The total number of steps to train for. Default: 50000")
    parser.add_argument('--n', '-n', type=str,
            help="The number of steps from where to start counting the next steps. Default: 0")
    args = parser.parse_args()

    args = vars(args)
    args = dict((k,v) for k,v in args.items() if v is not None)
    
    return args
