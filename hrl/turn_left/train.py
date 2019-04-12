#!/usr/bin/env python
from hrl.common.arg_extractor import get_args
from hrl.common.run_experiment import run_experiment

if __name__ == '__main__':
    # Run arg parser
    args = get_args()

    #env = CarRacing_turn

    # Run run experiment
    run_experiment(**args)
