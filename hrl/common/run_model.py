from copy import copy
from pdb import set_trace
import glob
import itertools
import os

from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import PPO2
import re
import numpy as np
import tensorboard_logger

from hrl.common.arg_extractor import get_load_args
from hrl.common.utils import create_experiment_folder,remove_experiment
from hrl.envs import env as environments

def load_model(
        experiment=None,
        folder='experiments', 
        weights=None,
        env='Base',
        full_path=None,
        policy=None,
        n_steps=None,
        tensorboard=False,
        tag=None,
        no_render=False,
        n_ep=None
        ):

    if policy != None:
        weights_loc = os.path.join("hrl/weights",policy)
        names = [name for name in os.listdir(weights_loc) if '.pkl' in name] # only pkl
        versions = [re.match(r'^(?:v)(\d+\.\d+)(?:_?)',i).group(1) for i in names] # capture v#.#_
        versions = [float(v) for v in versions] # Convert to float
        max_v = max(versions)
        w = [n for n in names if re.match(r'^v'+str(max_v),n) != None][0] # Getting max version name
        weights_loc = os.path.join(weights_loc,str(w))
    elif full_path != None:
        weights_loc = full_path
    else:
        if folder[-1] in '\\/':
            # remove \ from the end
            folder = folder[:-1]

        if weights is None:
            # Check what is the last weight
            weights_lst = [s for s in os.listdir('/'.join([folder,experiment])) if "weights_" in s]
            weights_lst = [s.replace('weights_','').replace('.pkl','') for s in weights_lst]
            if 'final' in weights_lst:
                weights = 'weights_final.pkl'
            else:
                weights_lst = [int(s) for s in weights_lst]
                weights = 'weights_' + str(max(weights_lst)) + '.pkl'
        
        weights_loc = '/'.join([folder,experiment,weights])
    print("**** Using weights",weights_loc)

    tb_logger = None
    if tensorboard:
        args = {'env':copy(env),
                'train_steps':n_steps,
                'weights':weights_loc}
        id,tb_logger,logs_folder,experiment_csv,experiment_folder =\
                create_experiment_folder(tag=tag,args=args)
        print("***** experiment is",experiment_folder)

    # Get env
    env = getattr(environments, env)(tensorboard_logger=tb_logger)
    if env.high_level and not no_render: env.auto_render = True
    env = DummyVecEnv([lambda: env])

    model = PPO2.load(weights_loc)
    model.set_env(env)

    obs = env.reset()
    done_count = 0
    reward = 0
    try:
        for current_step in itertools.count():
            action, _states = model.predict(obs)
            obs, rewards, dones, info = env.step(action)
            if not no_render:
                env.render()

            if any(dones) and tb_logger is None:
                print(reward)
            reward = env.get_attr("reward")[0]
            if any(dones) and n_ep is not None:
                done_count += 1
                if done_count % 20 == 0:
                    print("episode %i of %i" % (done_count,n_ep))
                if done_count >= n_ep:
                    break

            if n_steps is not None:
                if current_step % 1000 == 0:
                    print("steps %i of %i" % (current_step,n_step))
                if current_step >= n_steps:
                    break
    except KeyboardInterrupt:
        if tensorboard and input("Do you want to DELETE this experiment? (Yes/n) ") == "Yes":
            remove_experiment(experiment_folder, folder, experiment_csv, id)


if __name__ == '__main__':
    # Run arg parser
    args = get_load_args()

    # Run run experiment
    print(args)
    load_model(**args)
