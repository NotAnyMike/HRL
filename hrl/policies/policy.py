import numpy as np
from pdb import set_trace

from stable_baselines import PPO2
from stable_baselines.common.vec_env import DummyVecEnv

class Policy:
    def __init__(self,weights,id=None,max_steps=40):
        self.model = PPO2.load(weights)
        self.max_steps = max_steps
        self.n = 0
        self.id = id

    def __call__(self,env,state):
        action_rwrd = 0
        done  = False
        info  = {}
        self.n = 0

        obs = state
        env.add_active_policy(self.id)
        done = self._done(env)
        while self.n == 0 or (not self._done(env) and not done):
            action, _states = self.model.predict(obs)
            obs,rewards,done,info = self._raw_step(env,obs,action)

            action_rwrd += rewards
            self.n += 1
        env.remove_active_policy(self.id)

        return obs,action_rwrd,done,info

    def _raw_step(self,env,obs,action):
        obs, rewards, done, info = env.raw_step(action)
        return obs,rewards,done,info

    def _done(self,env):
        # The conditions are
        # 1. After n steps
        # 2. If it is outside the track
        # 3. If it is in objective
        # 4. If Timeout
        # 3 and 4 are complex, so I am not using them

        right = env.info['count_right'] > 0
        left  = env.info['count_left']  > 0

        done = False
        if self.n >= self.max_steps:
            done = True # 1
        elif (left|right).sum() == 0:
            done = True # 2

        return done


class HighPolicy(Policy):
    def __init__(self,weights,id):
        super(HighPolicy,self).__init__(weights,id=id)
        self.actions = []

    def _raw_step(self,env,obs,action):
        return self.actions[action](env,obs)


class Turn_left(Policy):
    def __init__(self):
        #super(Turn_left, self).__init__("hrl/weights/Turn_left/v1.0.pkl",id='TL')
        super(Turn_left, self).__init__("hrl/weights/Turn_left/v1.1_Final_of_exp73.pkl",id='TL')


class Turn_right(Policy):
    def __init__(self):
        super(Turn_right, self).__init__("hrl/weights/Turn_right/v1.0.pkl",id='TR')


class Take_center(Policy):
    def __init__(self):
        super(Take_center, self).__init__("hrl/weights/Take_center/v1.0.pkl",id='TC')


class Turn(HighPolicy):
    def __init__(self):
        #super(Turn, self).__init__("hrl/weights/Turn/v1.0.pkl",id='T')
        super(Turn, self).__init__("hrl/weights/Turn/v1.2.pkl",id='T')

        self.actions.append(Turn_left())
        self.actions.append(Turn_right())


class Y(Policy):
    def __init__(self):
        self.id = 'Y'
        self.turn = Turn()

    def __call__(self,env,state):
        obs = state
        env.add_active_policy(self.id)
        obs,rewards,done,info = self._raw_step(env,obs,0)
        env.remove_active_policy(self.id)

        return obs,rewards,done,info

    def _raw_step(self,env,obs,action):
        if action == 0:
            obs,reward,done,info = self.turn(env,obs)
        else:
            raise Exception("Action %i not implemented" % action)
        return obs,reward,done,info


class X(HighPolicy):
    def __init__(self):
        super(X,self).__init__("hrl/weights/X/v1.0.pkl",id='X')

        self.actions.append(Turn())
        self.actions.append(Take_center())


class Keep_lane(Policy):
    def __init__(self,max_steps=10):
        super(Keep_lane, self).__init__(
                "hrl/weights/Keep_lane/v1.0.pkl",
                id='KL',
                max_steps=max_steps,)
