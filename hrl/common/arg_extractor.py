import argparse
from pdb import set_trace

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def get_train_args():
    """ 
    Returns a namedtuple with arguments extracted from the command line.
    No default values are specified here to avoid double specification in
    run_experiment function. At the end no matter how the code gets executed
    run_experiment will always be run.

    :return: A namedtuple with arguments
    """
    parser = argparse.ArgumentParser(
        description='This will parse the argument and pass them \
                as parameters to th emian experiment function')

    parser.add_argument('--env', type=str, 
            help="The name of the class of the environment to run on")
    parser.add_argument('--train_steps', type=int,
            help="The total number of steps to train for.  \
                    Default: 1m")
    parser.add_argument('--n_steps', type=int,
            help="The number of steps to use in each training step")
    parser.add_argument('--save_interval', type=int,
            help="The model will be saved every number of steps  \
                    specified here, default: 10000")
    parser.add_argument('--weights', type=str,
            help="The position of the weights to load before \
                    continuing training")
    parser.add_argument('--n', '-n', type=int,
            help="The number of steps from where to start counting \
                    the next steps. Default: 0")
    parser.add_argument('--folder', type=str, 
            help="The folder where the experiment will be save.Defautl \
                    is 'experiments'")
    parser.add_argument('--env_num', type=int,
            help="The number of environements to use")
    parser.add_argument('--tag', type=str, 
            help="A tag to identify the experiment easier")
    parser.add_argument('--description','-d', type=str,
            help="A small description of the experiment")
    parser.add_argument('--not_save', action='store_true', 
            help="True means not saving the experiment")
    args = parser.parse_args()

    args = vars(args)
    args = dict((k,v) for k,v in args.items() if v is not None)
    
    return args

def get_load_args():
    """ 
    Returns a namedtuple with arguments extracted from the command line.
    No default values are specified here to avoid double specification in
    run_experiment function. At the end no matter how the code gets executed
    run_experiment will always be run.

    :return: A namedtuple with arguments
    """
    parser = argparse.ArgumentParser(
        description='This will parse the argument used in \
                loading a model and pass them as parameters to the loader function')

    parser.add_argument('--env', type=str, 
            help="The name of the class of the environment to run on, if not \
            specified Base is used")

    # In case path is complex
    parser.add_argument('--full_path', type=str,
            help="The full path of the weights, can be relative or absolute, \
            if this is given then -f,-e,-w are ignored")

    # To simplify loading policies
    parser.add_argument('--policy','-p', type=str,
            help="The name of the policy to load, it must have a folder in the \
            hrl/policies folder and one weight inside, the newest will be run,\
            if this option is given -f,-e,-w,--full_path are ignored")
    
    # To simplify loading weights
    parser.add_argument('--folder', '-f', type=str, 
            help="The folder of the experiments, by default it is 'experiments'")
    parser.add_argument('--experiment', '-e', type=str,
            help="The name of the folder of the experiment, the name of the \
            experiment, for example '2_base'")
    parser.add_argument('--weights', '-w', type=str,
            help="The name of the weights to load, If not specified we run the \
            last one or 'weights_final'")

    # Other parameters for checking performance/running exp
    parser.add_argument('--n_steps','-n', type=int,
            help="The number of steps to run for, if not specified then infinite")
    parser.add_argument('--n_ep', type=int,
            help="The number of steps to run for, if not specified then infinite")
    parser.add_argument('--tensorboard','-tb',action='store_true',
            help="A flag to register the score with tensorboard")
    parser.add_argument('--tag','-t', type=str,
            help="The tag for the folder in case of using tensorboard flag")
    parser.add_argument('--no_render', action='store_true',
            help="In case you want to log some info, but do not care about \
            rendering in screen")

    args = parser.parse_args()

    args = vars(args)
    args = dict((k,v) for k,v in args.items() if v is not None)

    if 'policy' in args:
        if 'full_policy' in args:del args['full_path']
        if 'folder' in args:     del args['folder']
        if 'experiment' in args: del args['experiment']
        if 'weights' in args:    del args['weights']
    elif "full_path" in args:
        if 'folder' in args:     del args['folder']
        if 'experiment' in args: del args['experiment']
        if 'weights' in args:    del args['weights']
    
    return args

def get_track_generator_args():
    parser = argparse.ArgumentParser(
        description='This will parse the argument used to generate tracks')

    parser.add_argument('--n', '-n', type=int, default=10000,
            help="The number of tracks to generate")
    parser.add_argument('--cpu', '-c', type=int, default=1,
            help="The number of cpus to use in parallel")
    args = parser.parse_args()

    return args

def get_env_args():
    parser = argparse.ArgumentParser(
        description='This will parse the argument used in \
                running a environment in play mode')

    parser.add_argument('--env', '-e', type=str, default="Base",
            help="The name of the class of the environment to run on, if not \
            specified Base is used")
    args = parser.parse_args()

    return args
