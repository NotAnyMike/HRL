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

    def _done(self,env,allow_outside=False):
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
        elif not allow_outside and (left|right).sum() == 0:
            done = True # 2

        return done


class HighPolicy(Policy):
    def __init__(self,weights,id,max_steps=10):
        super(HighPolicy,self).__init__(weights,id=id,max_steps=max_steps)

    def _raw_step(self,env,obs,action):
        return self.actions[action](env,obs)


class Turn_left(Policy):
    def __init__(self,id='TL',max_steps=4,v=None):
        if v == 1.0:
            """
            This version is used by Turn 1.2
            """
            w = "hrl/weights/Turn_left/v1.1_Final_of_exp73.pkl"
        elif v==1.2:
            """
            This version is re trained with x and also solves straight
            """
            w = "hrl/weights/Turn_left/v1.2_exp84_weights_final.pkl"
        elif v==1.3:
            """
            This version has more training
            """
            w = "hrl/weights/Turn_left/v1.3_exp111_weights_12601008.pkl"
        else:
            w = "hrl/weights/Turn_left/v1.3_exp111_weights_12601008.pkl"

        super(Turn_left, self).__init__(w,id=id,max_steps=max_steps)


class Turn_right(Policy):
    def __init__(self,id='TR',max_steps=4,v=None):
        if v==1.0:
            """
            This version is used by Turn 1.2
            """
            w = "hrl/weights/Turn_right/v1.0.pkl"
        elif v==1.2:
            """
            this version used x as well as straight directionals
            """
            w = "hrl/weights/Turn_right/v1.2_exp85_weights_final.pkl"
        elif v==1.3:
            """
            This version has more training
            """
            w = "hrl/weights/Turn_right/v1.3_exp112_weights_12601008.pkl"
        else:
            w = "hrl/weights/Turn_right/v1.3_exp112_weights_12601008.pkl"

        super(Turn_right, self).__init__(w,id=id,max_steps=max_steps)


class Take_center(Policy):
    def __init__(self,id='TC',max_steps=4,v=None):
        if v == 1.0:
            """
            faulty version with the same problem of Turn
            """
            w = "hrl/weights/Take_center/v1.0.pkl"
        else:
            w = "hrl/weights/Take_center/v1.0.pkl"

        super(Take_center, self).__init__(w,id=id,max_steps=max_steps)


class Turn(HighPolicy):
    def __init__(self,id='T',max_steps=0,v=None):
        self.actions = []
        if v==1.2:
            """
            this version is the one which Turn was not trained with X intersections,
            and then retrained with X intersections but X still behaves poorly
            """
            weights = "hrl/weights/Turn/v1.2.pkl"
            self.actions.append(Turn_left(v=1.0))
            self.actions.append(Turn_right(v=1.0))
        elif v==1.3:
            weights = "hrl/weights/Turn/v1.3_exp92_weights_final.pkl"
            self.actions.append(Turn_left(v=1.2))
            self.actions.append(Turn_right(v=1.2))
        elif v==1.4:
            weights = "hrl/weights/Turn/v1.4_exp147_weights_1347216.pkl"
            self.actions.append(Turn_left(v=1.3))
            self.actions.append(Turn_right(v=1.3))
        else:
            weights = "hrl/weights/Turn/v1.4_exp147_weights_1347216.pkl"
            self.actions.append(Turn_left())
            self.actions.append(Turn_right())

        super(Turn, self).__init__(
                weights,
                id=id,
                max_steps=max_steps)


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
    def __init__(self,id='X',max_steps=0,v=None):
        self.actions = []
        if v == 1.0:
            """
            Bad policy that does not handdle good because the 
            error with Turn in x intersections in an open world
            which is solved by updating Turn to 1.2
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
                id=id,
                max_steps=max_steps)


class Keep_lane(Policy):
    def __init__(self,id='KL',v=None,max_steps=4):
        if v == 1.0:
            w = "hrl/weights/Keep_lane/v1.0.pkl"
        else:
            w = "hrl/weights/Keep_lane/v1.0.pkl"
        super(Keep_lane, self).__init__(
                w,
                id=id,
                max_steps=max_steps,)


class Change_to_right(Policy):
    def __init__(self,v=None,max_steps=4):
        if v == 1.0:
            w = "hrl/weights/CRight/v1.0_exp83_weights_final.pkl"
        elif v ==1.05:
            """
            it has 6m of steps more than v1.1 but is not better
            """
            w = "hrl/weights/CRight/v1.05_exp115_weights_2402304.pkl"
        elif v==1.1:
            w = "hrl/weights/CRight/v1.1_exp109_weights_final.pkl"
        else:
            w = "hrl/weights/CRight/v1.1_exp109_weights_final.pkl"

        super(Change_to_right, self).__init__(
                w,
                id='CR',
                max_steps=max_steps,)


class Change_to_left(Policy):
    def __init__(self,v=None,max_steps=4):
        if v == 1.0:
            w = "hrl/weights/CLeft/v1.0_exp82_weights_final.pkl"
        elif v==1.2:
            w = "hrl/weights/CLeft/v1.1_exp108_weights_final.pkl"
        elif v==1.3:
            w = "hrl/weights/CLeft/v1.2_exp127_weights_final.pkl"
        else:
            w = "hrl/weights/CLeft/v1.2_exp127_weights_final.pkl"
            
        super(Change_to_left, self).__init__(
                w,
                id='CL',
                max_steps=max_steps,)


class Change_lane(HighPolicy):
    def __init__(self,id='CLane', v=None, max_steps=0):
        if v == 1.0:
            w = "hrl/weights/Change_lane/v1.0_exp86_weights_final.pkl"
        elif v==1.1:
            w = "hrl/weights/Change_lane/v1.2_exp113_weights_final.pkl"
        elif v==2.0:
            """
            This version does avoid obstacles
            """
            w = "hrl/weights/Change_lane/v2.0_135_CL_B_weights_3500928.pkl"
        else:
            w = "hrl/weights/Change_lane/v2.0_135_CL_B_weights_3500928.pkl"

        self.actions = []
        self.actions.append(Change_to_left())
        self.actions.append(Change_to_right())

        super(Change_lane,self).__init__(w,id=id,max_steps=max_steps)


class NWOO(HighPolicy):
    def __init__(self,id='NWOO',max_steps=0, v=None):
        self.actions = []
        if v == 1.0:
            """
            this version is suboptimal, relies 100% in Y and forgets about X,
            it can be due to the unbalance between the number of tracks chosen
            with X and Y intersections
            """
            w = "hrl/weights/NWOO/v1.0_exp93-NWOO-v1.1_weights_final.pkl"
            self.actions.append(Keep_lane(v=1.0))
            self.actions.append(X(v=1.0))
            self.actions.append(Y(v=1.0))
        else:
            w = "hrl/weights/NWOO/v1.0_exp93-NWOO-v1.1_weights_final.pkl"
            self.actions.append(Keep_lane())
            self.actions.append(X())
            self.actions.append(Y())

        super(NWOO,self).__init__(w,id=id,max_steps=max_steps)


class Recovery_delayed(Policy):
    def __init__(self,id='De',v=None,max_steps=10):
        w = "hrl/weights/De/v1.0_exp95_weights_final.pkl"
        
        super(Recovery_delayed,self).__init__(w,id=id,max_steps=max_steps)

    def _done(self,env,allow_outside=True):
        return super(Recovery_delayed, self)._done(env,allow_outside=allow_outside)


class Recovery_direct(Policy):
    def __init__(self,id='D',v=None,max_steps=10):
        w = "hrl/weights/D/v1.0_exp96_weights_final.pkl"
        
        super(Recovery_direct,self).__init__(w,id=id,max_steps=max_steps)

    def _done(self,env,allow_outside=True):
        return super(Recovery_direct, self)._done(env,allow_outside=allow_outside)


class Recovery(HighPolicy):
    def __init__(self,id='R',v=None, max_steps=0):
        self.actions = []
        self.actions.append(Recovery_direct())
        self.actions.append(Recovery_delayed())

        w = "hrl/weights/Recovery/v0.2_exp98_weights_final.pkl"

        super(Recovery,self).__init__(w,id=id,max_steps=max_steps)


class NWO(HighPolicy):
    def __init__(self,id='NWO',v=None,max_steps=0):
        self.actions = []
        self.actions.append(Keep_lane())
        self.actions.append(Change_lane())

        w = "hrl/weights/NWO/v0.1_exp99_weights_1028896.pkl"

        super(NWO,self).__init__(w,id=id,max_steps=max_steps)
