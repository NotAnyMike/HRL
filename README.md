# Hierarchical Reinforcement Learning

This repo replicates this hierarchy for a theoretical AV

![hierachy](img/hierarchy.png)

## Installation

this repo mainly uses two big repos, `stable-baselines` and `gym2`.

0. If you prefer, create a new conda environemnt
1. Install `stable-baselines` with `pip`
2. Install the custom version of gym called `gym2` which integrates an improved env `CarRacing`
3. Install pandas with `pip install pandas`


## How to run experiments

There are two ways of running experiments which support logging and saving of all the experiment important information, the first one is using the function `run_experiment` in the `source` folder. The second way to run an experiment is using direclty the console. All the files will be saved in a folder inside the `folder` folder and will be named `<id>_<tag>` where `<id>` is the id in the id experiments table and `<tag>` is the tag argument.

### Using `run_experiment`

The function `run_experiment` takes several different arguments, here it is a table of a summary of all of them

| Argument | Type of value | Default value | Description |
| --- | --- | --- | --- |
| `save` | bool | True | Whether or not save and log the experiment |
| `folder` | str | 'experiments' | The name of the folder where to save everything, cannot be empty |

### Using `run_model`

The only important argument to give is the name of the experiment, e.g. `-e 5`, that will run the last saved weights with the base environemnt.

---

## IDEAS

### Regarding the baseline

To improve it try:
- [ ] Time out reduce to 1 or less
- [ ] Max reward per episode to 5, in order to make it faster, maybe a number betwen 1 and 5 is better

### Regarding the left policy

To improve it try:
- [ ] Using multiple environements at the same time
- [ ] Max reward = 1
- [ ] Initialize the weights with the baseline
