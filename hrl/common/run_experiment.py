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
        save_interval=10000,
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
        id = df.index[-1]

        # Creating folder for experiment
        experiment_folder = '/'.join([folder,str(df.index[-1])])
        os.makedirs(experiment_folder)

        del args
        del df


    # Start running experiment
    print("########################")
    print("# RUNNING EXPERIMENT   #")
    print("# ID: %i " % id) 
    print("########################")

    logs_folder = experiment_folder + '/logs'
    logger = Logger(logs_folder+"/extra")

    #env = SubprocVecEnv([lambda: env for i in range(1)]) # Working but
    env = DummyVecEnv([env])
    model = PPO2(
                CnnPolicy, 
                env,
                verbose=0,
                tensorboard_log=logs_folder,
                max_grad_norm=100,
                n_steps=200,
                policy_kwargs={'data_format':'NCHW'},
            )

    # Set key functions
    show_hide = Show_hide(model,experiment_folder)
    for tmp_env in env.envs:
        tmp_env.set_press_fn(show_hide)

    # set bar
    callback = Callback(
            logger=logger,
            train_steps=train_steps,
            n=n,
            experiment_folder=experiment_folder,
            save_interval=save_interval,
            id=id,
            )

    print("\n############ STARTING TRAINING ###########\n")
    with tqdm.tqdm(total=train_steps, leave=True) as bar:
        callback.set_bars(bar)
        model.learn(
                total_timesteps=train_steps,
                callback=callback,
                )

    model.save(experiment_folder+"/weights_final")

class Show_hide:
    def __init__(self,model,experiment_folder="experiments/"):
        self.model = model
        self.experiment_folder = experiment_folder

    def __call__(self,k, mod):
        if k==key.S: # S from show
            self.model.render = not self.model.render
        if k==key.T: # T from T screnshot
            self.model.env.envs[0].screenshot(self.experiment_folder)

class Callback:
    def __init__(self, logger,train_steps,n,experiment_folder,
            save_interval, id):

        self.last_step = 0

        self.logger = logger
        self.n = n
        self.train_steps = train_steps
        self.experiment_folder = experiment_folder
        self.save_interval = save_interval 
        self.id = id

    def set_bars(self, global_bar):
        self.global_bar = global_bar

    def __call__(self, local_vars, global_vars):
        current_step = int(self.n + local_vars['self'].num_timesteps)
        self.global_bar.update(current_step - self.last_step)

        # TODO Print the name of experiment
        self.global_bar.set_description("Training | ID: %i | fps: %i" \
                % (self.id,int(local_vars['fps'])))

        if self.save_interval > 0:
            if current_step - self.last_step > self.save_interval:
                local_vars['self'].save(self.experiment_folder + '/weights_'+str(current_step))

        #set_trace()

        # TODO
        # Reward also because the normal logger does not log every episode
        # Log speed
        # Log angle
        # Log actions taken
        # Log num steps

        self.last_step = current_step
