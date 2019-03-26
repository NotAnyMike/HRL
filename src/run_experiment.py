import os
from pdb import set_trace
from copy import deepcopy

import pandas as pd
import numpy as np

from arg_extractor import get_args

def run_experiment(
        save=True, 
        folder='experiments', 
        weights_location=None,
        tag=None,
        patience=20,
        ):
    """
    save 
    folder
    tag
    weights_location
    description

    patience
    """
    # saving args
    args = deepcopy(locals())
    
    # Check if folder exists and is a valid name
    if save:
        folder = folder.replace(' ', '_')
        if os.path.exists(folder):
            print(" - Folder for experiments found")
        else:
            print(" - Creating folder for experiments")
            os.makedirs(folder)

        # Load cvs of experiments
        experiment_csv = '/'.join([folder, "experiments.csv"])
        if os.path.isfile(experiment_csv):
            print(" - Loading experiments.csv file")
            df = pd.read_csv(experiment_csv)
        else:
            print(" - experiments.csv not found, creating one")
            df = pd.DataFrame(columns=args.keys())
            df.to_csv(experiment_csv)

        df = df.append(args, ignore_index=True)
        df.to_csv(experiment_csv)

    pass

if __name__ == '__main__':
    # Run arg parser
    args = vars(get_args())
    args = dict((k,v) for k,v in args.items() if v is not None)

    # Run run experiment
    run_experiment(**args)
