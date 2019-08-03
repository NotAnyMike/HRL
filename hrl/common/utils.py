from tensorboard_logger import Logger
import os
import pandas as pd
import shutil
from pdb import set_trace

def create_experiment_folder(folder='experiments',tag=None,args=None):
    if folder is None: folder = 'experiments'

    if os.path.exists(folder + '/to_delete'):
        shutil.rmtree(folder + '/to_delete')

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
        df = pd.read_csv(experiment_csv, index_col=0)
    else:
        print(" - experiments.csv not found, creating one")
        df = pd.DataFrame(columns=args.keys())
        df.to_csv(experiment_csv)

    #df = df.append(args, ignore_index=True)
    df.loc[df.index.max()+1] = pd.Series(args)
    df.to_csv(experiment_csv)
    id = df.index[-1]

    # Creating folder for experiment
    if tag is None: 
        experiment_folder = '/'.join([folder,str(df.index[-1])])
    else: 
        experiment_folder = '/'.join([folder,str(df.index[-1])+'_'+tag])
    os.makedirs(experiment_folder)

    logs_folder = experiment_folder + '/logs'
    logger = Logger(logs_folder+"/extra")

    del df

    return id, logger, logs_folder, experiment_csv, experiment_folder

def remove_experiment(experiment_folder, folder, experiment_csv, id):
    df = pd.read_csv(experiment_csv, index_col=0)
    df.drop(id,inplace=True)
    df.to_csv(experiment_csv)

    os.rename(experiment_folder, folder + '/to_delete/')
