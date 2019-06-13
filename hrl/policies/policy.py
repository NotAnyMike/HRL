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
    def __init__(self,weights,id,max_steps=10):
        super(HighPolicy,self).__init__(weights,id=id,max_steps=max_steps)
        self.actions = []

    def _raw_step(self,env,obs,action):
        return self.actions[action](env,obs)


class Turn_left(Policy):
    def __init__(self,v=None):
        #super(Turn_left, self).__init__("hrl/weights/Turn_left/v1.0.pkl",id='TL')
        if v == 1.0:
            """
            This version is used by Turn 1.2
            """
            w = "hrl/weights/Turn_left/v1.1_Final_of_exp73.pkl"
        else:
            w = "hrl/weights/Turn_left/v1.1_Final_of_exp73.pkl"

        super(Turn_left, self).__init__(w,id='TL')


class Turn_right(Policy):
    def __init__(self,v=None):
        if v==1.0:
            """
            This version is used by Turn 1.2
            """
            w = "hrl/weights/Turn_right/v1.0.pkl"
        else:
            w = "hrl/weights/Turn_right/v1.0.pkl"

        super(Turn_right, self).__init__(w,id='TR')


class Take_center(Policy):
    def __init__(self,v=None):
        if v == 1.0:
            """
            faulty version with the same problem of Turn
            """
            w = "hrl/weights/Take_center/v1.0.pkl"
        else:
            w = "hrl/weights/Take_center/v1.0.pkl"

        super(Take_center, self).__init__(w,id='TC')


class Turn(HighPolicy):
    def __init__(self,v=None):
        #super(Turn, self).__init__("hrl/weights/Turn/v1.0.pkl",id='T')
        if v==1.2:
            """
            this version is the one which Turn was not trained with X intersections,
            and then retrained with X intersections but X still behaves poorly
            """
            weights = "hrl/weights/Turn/v1.2.pkl"
            self.actions.append(Turn_left(v=1.0))
            self.actions.append(Turn_right(v=1.0))
        else:
            weights = "hrl/weights/Turn/v1.2.pkl"
            self.actions.append(Turn_left())
            self.actions.append(Turn_right())

        super(Turn, self).__init__(
                weights,
                id='T',
                max_steps=0)


class Y(Policy):
    def __init__(self,v=None):
        self.id = 'Y'
        if v==1.0:
            """
            this version is a poorly one in an open environment
            """
            self.turn = Turn(v=1.2)
        else:
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
    def __init__(self,v=None):
        if v == 1.0:
            """
            Bad policy that does not handdle good because the 
            error with Turn in x intersections in an open world
            """
            w = "hrl/weights/X/v1.0.pkl"
            self.actions.append(Turn(v=1.0))
            self.actions.append(Take_center(v=1.0))
        else:
            w = "hrl/weights/X/v1.0.pkl"
            self.actions.append(Turn())
            self.actions.append(Take_center())

        super(X,self).__init__(
                w,
                id='X',
                max_steps=0)


class Keep_lane(Policy):
    def __init__(self,v=None,max_steps=10):
        super(Keep_lane, self).__init__(
                "hrl/weights/Keep_lane/v1.0.pkl",
                id='KL',
                max_steps=max_steps,)


class Change_to_right(Policy):
    def __init__(self,v=None,max_steps=50):
        super(Change_to_right, self).__init__(
                "hrl/weights/CRight/v1.0_exp83_weights_final.pkl",
                id='CR',
                max_steps=max_steps,)


class Change_to_left(Policy):
    def __init__(self,v=None,max_steps=50):
        super(Change_to_left, self).__init__(
                "hrl/weights/CLeft/v1.0_exp82_weights_final.pkl",
                id='CL',
                max_steps=max_steps,)
