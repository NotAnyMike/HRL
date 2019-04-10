import argparse

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def get_args():
    """ Returns a namedtuple with arguments extracted from the command line.
    :return: A namedtuple with arguments
    """
    parser = argparse.ArgumentParser(
        description='This will parse the argument and pass them as parameters to th emian experiment function')

    parser.add_argument('--folder', type=str, help="")
    parser.add_argument('--patience', type=int,
                        help='The number of epochs to wait for early stopping')
    args = parser.parse_args()
    return args
