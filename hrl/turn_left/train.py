#!/usr/bin/env python
from hrl.common.arg_extractor import get_args
from hrl.common.run_experiment import run_experiment
from hrl.turn_left.env import CarRacing_turn

if __name__ == '__main__':
    # Run arg parser
    args = get_args()

    env = CarRacing_turn

    # Run run experiment
    run_experiment(env=env, **args)
