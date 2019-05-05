import numpy as np

from stable_baselines import PPO2

class Policy:
    def __init__(self,weights,env):
        self.model = PPO2.load(weights)
        setf.model.set_env(env)
        self.max_steps = 90
        self.n = 0

    def __call__(self,state):
        action_rwrd = 0
        done  = False
        info  = {}
        self.n = 0

        obs = state
        while not self._done():
            action, _states = model.predict(obs)
            obs,rewards,done,info = env.raw_step(action)

            action_rwrd += rewards
            self.n += 1

        return obs,action_rwrd,done,info

    def _done(self):
        # The conditions are
        # 1. After n steps
        # 2. If it is outside the track
        # 3. If it is in objective
        # 4. If Timeout
        # 3 and 4 are complex, so I am not using them

        right = self.env.info['count_right'] > 0
        left  = self.env.info['count_left']  > 0

        done = False
        if self.n >= self.max_steps:
            done = True # 1
        elif (left|right).sum() == 0:
            done = True # 2

        return done

class Turn_left(Policy):
    def __init__(self,env):
        super(Turn_left, self).__init__("hrl/weights/Turn_left/v1.0.pkl", env)

class Turn_right(Policy):
    def __init__(self,env):
        super(Turn_left, self).__init__("hrl/weights/Turn_right/v1.0.pkl", env)
