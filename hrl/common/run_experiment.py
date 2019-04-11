import os
from pdb import set_trace
from copy import deepcopy

import pandas as pd
import numpy as np
import tqdm
from tensorboard_logger import Logger
from stable_baselines.common.policies import CnnPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines.common.vec_env import SubprocVecEnv
from stable_baselines import PPO2
from pyglet.window import key

from hrl.common.arg_extractor import get_args

def run_experiment(
        save=True, 
        folder='experiments', 
        weights_location=None,
        tag=None,
        env=None,
        n=0,
        save_interval=0,
        train_steps=500000,
        ):
    """
    save 
    folder
    tag
    weights_location
    description
    env
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
            df = pd.read_csv(experiment_csv, index_col=0)
        else:
            print(" - experiments.csv not found, creating one")
            df = pd.DataFrame(columns=args.keys())
            df.to_csv(experiment_csv)

        df = df.append(args, ignore_index=True)
        df.to_csv(experiment_csv)

        # Creating folder for experiment
        experiment_folder = '/'.join([folder,str(df.index[-1])])
        os.makedirs(experiment_folder)

        del args


    # Start running experiment
    print("########################")
    print("# RUNNING EXPERIMENT   #")
    print("# ID: %i " % df.index[-1]) 
    print("########################")

    logs_folder = experiment_folder + '/logs'
    logger = Logger(logs_folder)

    #env = SubprocVecEnv([lambda: env for i in range(1)]) # Working but
    env = DummyVecEnv([env])
    model = PPO2(
                CnnPolicy, 
                env,
                verbose=1, 
                tensorboard_log=logs_folder,
                max_grad_norm=100,
                n_steps=200,
                policy_kwargs={'data_format':'NCHW'},
            )

    # Set key functions
    show_hide = Show_hide(model)
    for tmp_env in env.envs:
        tmp_env.set_press_fn(show_hide)

    # set bar
    callback = Callback(
            logger=logger,
            train_steps=train_steps,
            n=0)
    with tqdm.tqdm(total=train_steps, leave=True) as bar:
        callback.set_bars(bar)
        model.learn(
                total_timesteps=train_steps,
                callback=callback,
                )

    model.save(experiment_folder+"_final")

class Show_hide:
    def __init__(self,model):
        self.model = model

    def __call__(self,k, mod):
        if k==key.S:
            self.model.render = not self.model.render

class Callback:
    def __init__(self, logger,train_steps, n):
        self.logger = logger
        self.n = n
        self.train_steps = train_steps

    def set_bars(self, global_bar):
        self.global_bar = global_bar

    def __call__(self, local_vars, global_vars):
        self.global_bar.update(1)
        self.n += 1
        self.global_bar.set_description("Training: steps %i / %i" % (self.n,self.train_steps))
