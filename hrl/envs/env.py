from copy import copy, deepcopy
import multiprocessing as mp
import sys

from gym.envs.box2d import CarRacing
from gym.envs.box2d.car_racing import play, default_reward_callback, TILE_NAME, SOFT_NEG_REWARD, HARD_NEG_REWARD, WINDOW_W, WINDOW_H, TRACK_WIDTH
from gym import spaces
from pdb import set_trace
from pyglet import gl
import numpy as np

from hrl.common.arg_extractor import get_env_args
from hrl.envs import env as environments
from hrl.policies.policy import Turn_left as Left_policy
from hrl.policies.policy import Turn_right as Right_policy
from hrl.policies.policy import Turn as Turn_policy
from hrl.policies.policy import Take_center as Take_center_policy
from hrl.policies.policy import Keep_lane as Keep_lane_policy
from hrl.policies.policy import Y as Y_policy
from hrl.policies.policy import X as X_policy
from hrl.policies.policy import Change_to_left as Change_to_left_policy
from hrl.policies.policy import Change_to_right as Change_to_right_policy
from hrl.policies.policy import Change_lane as Change_lane_policy
from hrl.common.visualiser import PickleWrapper, Plotter, worker

class Base(CarRacing):
    def __init__(self, 
            reward_fn=default_reward_callback,
            max_time_out=2.0,
            max_step_reward=1,
            auto_render=False,
            high_level=False,
            id='Nav',
            load_tracks_from="tracks",
            *args,
            **kwargs
            ):
        super(Base,self).__init__(
                allow_reverse=False, 
                grayscale=1,
                show_info_panel=False,
                verbose=0,
                discretize_actions="hard",
                num_tracks=2,
                num_lanes=2,
                num_lanes_changes=0,
                num_obstacles=100,
                max_time_out=max_time_out,
                frames_per_state=4,
                max_step_reward=max_step_reward,
                reward_fn=reward_fn,
                random_obstacle_x_position=False,
                random_obstacle_shape=False,
                load_tracks_from=load_tracks_from,
                *args,
                **kwargs
                )
        self.high_level = high_level
        self.visualiser_process = None
        self.ID = id
        self.active_policies = set([self.ID])
        self.stats = {}
    
    def _key_press(self,k,mod):
        # to avoid running a process inside a daemon
        if mp.current_process().name == 'MainProcess':
            from pyglet.window import key
            if k == key.B: # B from dashBoard
                if self.visualiser_process == None:
                    # Create visualiser
                    self.connection, child_conn = mp.Pipe()
                    to_pickle = lambda: Plotter()
                    args = (PickleWrapper(to_pickle),child_conn)
                    self.ctx = mp.get_context('spawn')
                    self.visualiser_process = self.ctx.Process(
                            target=worker,
                            args=args,
                            daemon=True,)
                    self.visualiser_process.start()

                    self.connection.send(("add_active_policies",
                            [[self.active_policies],{}]))
                else:
                    self.visualiser_process.terminate()
                    del self.visualiser_process
                    del self.connection
                    del self.ctx
                    self.visualiser_process = None

        super(Base,self)._key_press(k,mod)

    def _key_release(self,k,mod):
        super(Base,self)._key_release(k,mod)

    def add_active_policy(self,policy_name):
        self.active_policies.add(policy_name) 
        if self.visualiser_process is not None:
            self.connection.send(("add_active_policy", [[policy_name],{}]))

    def remove_active_policy(self,policy_name):
        self.active_policies.remove(policy_name)
        if self.visualiser_process is not None:
            self.connection.send(("remove_active_policy", [[policy_name],{}]))

    def _ignore_obstacles(self):
        '''
        Calling this will make every reward function to ignore obstacles,
        this, has to be called every time a reward function is called, at 
        the beginning of the function or before it.
        '''
        self.obstacle_contacts['count_delay'] = 0
        self.obstacle_contacts['count'] = 0

    def _check_if_close_to_intersection(self,direction=1):
        close = False
        for tile in np.where(
                (self.info['count_left'] > 0) \
                    | (self.info['count_right'] > 0))[0]:
            if self._is_close_to_intersection(tile,direction=direction):
                close = True
                break
        return close

    def _is_close_to_intersection(self,tile_id,spaces=8,direction=1):
        '''
        Return true if the tile_id tile is direction spaces from
        and intersection

        direction = {1,0,-1}, going with the flow or agains it, -
        means both directions
        '''
        intersection_tiles = self.get_close_intersections(
                tile_id=tile_id,spaces=spaces,direction=direction)
        return len(intersection_tiles) > 0

    def get_close_intersections(self,tile_id,spaces=8,direction=1):
        '''
        Returns a set of the index of the tiles that are space close in
        the direction direction of tile_id
        '''
        if spaces > 0 and direction in [1,0,-1]:
            track_id = self.info[tile_id]['track']
            track_len = len(self.tracks[track_id])
            other_track_len  = sum(self.info['track'] < track_id)

            tile_id_rel =  tile_id - other_track_len

            candidates = [] # If tile_id is intersection, this return False
            stop_0, stop_1 = False, False
            for i in range(spaces):
                if direction in [1,0] and not stop_1:
                    possible_candidate = tile_id_rel + 1 + i
                    possible_candidate %= track_len
                    possible_candidate += other_track_len
                    candidates.append(possible_candidate)
                    if self.info[possible_candidate]['intersection_id'] != -1:
                        stop_1 = True
                if direction in [-1,0] and not stop_0:
                    possible_candidate = tile_id_rel - 1 - i
                    possible_candidate %= track_len
                    possible_candidate += other_track_len
                    candidates.append(possible_candidate)
                    if self.info[possible_candidate]['intersection_id'] != -1:
                        stop_0 = True

            candidates = set(candidates)
            positive_candidates = candidates.intersection(
                    np.where(self.info['intersection_id'] != -1)[0])

            return positive_candidates
        else:
            raise ValueError("Check the attributes used in \
                    _is_close_to_intersection")

    def _render_center_arrow(self):
        # Arrow
        gl.glBegin(gl.GL_TRIANGLES)
        gl.glColor4f(0.7,0,0,1)
        gl.glVertex3f(WINDOW_W//2, WINDOW_H-40,0)
        gl.glVertex3f(WINDOW_W//2-40,WINDOW_H-40-40,0)
        gl.glVertex3f(WINDOW_W//2+40,WINDOW_H-40-40,0)
        gl.glEnd()

        # Body
        gl.glBegin(gl.GL_QUADS)
        gl.glColor4f(0.7,0,0,1)
        gl.glVertex3f(WINDOW_W//2+15,WINDOW_H-80,0)
        gl.glVertex3f(WINDOW_W//2+15,WINDOW_H-80-50,0)
        gl.glVertex3f(WINDOW_W//2-15,WINDOW_H-80-50,0)
        gl.glVertex3f(WINDOW_W//2-15,WINDOW_H-80,0)
        gl.glEnd()

    def _render_side_arrow(self,lat_dir,long_dir):
        '''
        lat_dir is left or right
        long_dir is 1 or -1
        '''
        d = 1 if lat_dir == 'right' else 0
        long_dir = (-1)**d

        # Arrow
        gl.glBegin(gl.GL_TRIANGLES)
        gl.glColor4f(0.7,0,0,1)
        gl.glVertex3f(WINDOW_W*d+long_dir*20, WINDOW_H-80,0)
        gl.glVertex3f(WINDOW_W*d+long_dir*100,WINDOW_H-80-40,0)
        gl.glVertex3f(WINDOW_W*d+long_dir*100,WINDOW_H-80+40,0)
        gl.glEnd()

        # Body
        gl.glBegin(gl.GL_QUADS)
        gl.glColor4f(0.7,0,0,1)
        gl.glVertex3f(WINDOW_W*d+long_dir*100,WINDOW_H-80+15,0)
        gl.glVertex3f(WINDOW_W*d+long_dir*150,WINDOW_H-80+15,0)
        gl.glVertex3f(WINDOW_W*d+long_dir*150,WINDOW_H-80-15,0)
        gl.glVertex3f(WINDOW_W*d+long_dir*100,WINDOW_H-80-15,0)
        gl.glEnd()


class High_level_env_extension():
    def __init__(self,high_level=True,*args,**kwargs):
        super(High_level_env_extension,self).__init__(
                high_level=high_level,*args,**kwargs)

        self.action_space = spaces.Discrete(len(self.actions))

    def raw_step(self,action):
        # Normal step 
        return super(High_level_env_extension,self).step(action)

    def step(self,action):
        if action is None:
            state, reward, done, info = self.raw_step(None)
        else:
            # execute transformed action
            state, reward, done, info = self.actions[action](self,self.state)

        return state, reward, done, info


class Turn_side(Base):
    def __init__(self, 
            id='T', 
            high_level=False, 
            max_time_out=1.0, 
            max_step_reward=10, 
            allow_outside=False,
            reward_function=None,
            *args, **kwargs):

        def reward_fn(env):
            reward = -SOFT_NEG_REWARD
            done = False

            predictions_id = [id for l in env._next_nodes for id in l.keys() ]
            right_old = env.info['count_right_delay'] > 0
            left_old  = env.info['count_left_delay']  > 0
            not_visited = env.info['visited'] == False
            right = env.info['count_right'] > 0
            left  = env.info['count_left']  > 0
            track0 = env.info['track'] == 0
            track1 = env.info['track'] == 1

            if env.goal_id in np.where(right_old|left_old)[0]:
                # If in objective
                reward += 10
                done = True
            else:
                reward,done = env.check_outside(reward,done)
                reward,done = env.check_timeout(reward,done)
                reward,done = env.check_unvisited_tiles(reward,done)
                
                # if still in the prediction set
                if not done and len(list( set(predictions_id) & set(\
                        np.where(not_visited & (right_old|left_old))[0]))) > 0:
        
                    # To allow changes of lane in intersections without lossing points
                    if (left_old & right_old & track0).sum() > 0 and (left_old & right_old & track1).sum() > 0:
                        factor = 2
                    elif (left_old & right_old & track1).sum() > 0 and (((left_old | right_old) & track0).sum() == 0) :
                        factor = 2
                    elif (left_old & right_old & track0).sum() > 0 and (((left_old | right_old) & track1).sum() == 0) :
                        factor = 2
                    else:
                        factor = 1 
                        
                    reward += 1 / factor

            # Cliping reward per episode
            full_reward = reward
            reward = np.clip(
                    reward, env.min_step_reward, env.max_step_reward)

            env.info['visited'][left_old | right_old] = True
            env.info['count_right_delay'] = env.info['count_right']
            env.info['count_left_delay']  = env.info['count_left']
            
            return reward,full_reward,done

        super(Turn_side,self).__init__(
                reward_fn=reward_fn,
                high_level=high_level, 
                allow_outside=allow_outside,
                max_step_reward=max_step_reward,
                id=id,
                *args, 
                **kwargs,
                )
        self.goal_id = None
        self.new = True
        self._reward_fn_side = reward_fn
        self.tracks_df = self.tracks_df[(self.tracks_df['t']) | (self.tracks_df['x'])]

    def _choice_random_track_from_file(self):
        if np.random.uniform() >= 0.5:
            idx = np.random.choice(self.tracks_df[self.tracks_df['x']].index)
        else:
            idx = np.random.choice(self.tracks_df[self.tracks_df['t']].index)
        return idx

    def update_contact_with_track(self):
        self.update_contact_with_track_side()

    def update_contact_with_track_side(self):
        # Only updating it if still in prediction

        not_visited = self.info['visited'] == False
        right = self.info['count_right'] > 0
        left  = self.info['count_left']  > 0
        predictions_id = [id for l in self._next_nodes for id in l.keys() ]

        # Intersection of the prediction and the set that you are currently in
        if len(list( set(predictions_id) & set(\
                np.where(right|left)[0]))) > 0:
            self.last_touch_with_track = self.t
            #super(Turn_side,self).update_contact_with_track()

    def _generate_predictions_side(self,filter):
        Ok = True
        tiles_before = 8
        idx = np.random.choice(np.where(filter)[0])

        idx_relative = idx - (self.info['track'] < self.info[idx]['track']).sum()
        if self.info[idx]['end']:
            direction = +1
        elif self.info[idx]['start']:
            direction = -1
        else:
            direction = 1 if np.random.random() < 0.5 else -1
        idx_relative_new = (idx_relative - direction*tiles_before) % len(self.tracks[self.info[idx]['track']])

        idx_new = idx - idx_relative + idx_relative_new
        _,beta,x,y = self._get_rnd_position_inside_lane(idx_new,direction=direction)            

        angle_noise = np.random.uniform(-1,1)*np.pi/12
        beta += angle_noise

        self.place_agent([beta,x,y])
        speed = np.random.uniform(0,100)
        self.set_speed(speed)

        # adding nodes before coming to the intersection
        self._next_nodes = []
        for i in range(tiles_before):
            idx_tmp = (idx_relative - direction*i) % len(self.tracks[self.info[idx]['track']])
            idx_tmp = idx - idx_relative + idx_tmp
            self._next_nodes.append({idx_tmp: {0:-direction,1:-direction}})

        inter = self.understand_intersection(idx,direction)
        
        def get_avg_d_of_segment(idx,direction):
            """
            Calculate the distance of a segment (between two intersections) to
            the center of the map.
            """
            # get the segments
            intersection_id = self.info[idx]['intersection_id']
            track_id = self.info[idx]['track']
            intersections_nodes = np.where((self.info['intersection_id'] != -1) & (self.info['track'] == track_id))[0]
            tmp_pos = np.where(intersections_nodes == idx)[0][0]
            if direction == 1:
                end = idx
                start = intersections_nodes[(tmp_pos-1)%len(intersections_nodes)]
            else:
                start = idx
                end = intersections_nodes[(tmp_pos+1)%len(intersections_nodes)]

            ids = []
            if start > end:
                ids = []
                len_other_tracks = (self.info['track'] < track_id).sum()
                len_track = len(self.tracks[track_id])
                start = start-len_other_tracks
                end   = end-len_other_tracks
                while start != end:
                    ids.append(len_other_tracks+start)
                    start = (start+1)%len_track
                segment = self.track[ids][:,1,2:]
            else:
                ids = list(range(start,end))
                segment = self.track[ids,1,2:]
            avg_d = np.mean(np.linalg.norm(segment, axis=1))
            return avg_d,ids

        ######## adding nodes after the intersection
        # get avg distance from the origin
        intersection_id = self.info[idx]['intersection_id']
        current_track = self.info[idx]['track']
        node_id_main_track = np.where((self.info['intersection_id'] == intersection_id) & (self.info['track'] == 0))[0][0]
        node_id_second_track = np.where((self.info['intersection_id'] == intersection_id) & (self.info['track'] == 1))[0]

        if len(node_id_second_track) == 0:
            # There is no other point in the second track
            return False

        if direction == 1:
            node_id_second_track = node_id_second_track[-1]
        else:
            node_id_second_track = node_id_second_track[0]

        avg_main_track,ids= get_avg_d_of_segment(node_id_main_track, direction)
        # Debugging purposes
        #for i in ids:
        #   self._next_nodes.append({i:{0:direction,1:direction}})

        avg_second_track,ids = get_avg_d_of_segment(node_id_second_track, direction)
        # Debugging purposes
        #for i in ids:
        #    self._next_nodes.append({i:{0:direction,1:direction}})

        # The direction of lane
        flow = self._flow
        if current_track == 0: 
            #flow *= -1 # TODO original code, check it there is no problem
            # This line below was added to avoid having the filter in X class only
            # searching for x intersections in second track, this in theory
            # will allow correct renderisation of predictions and arrows for any
            # filter, but it is necessary to check if this code does not break
            # Turn, Turn_left and right, if it does not remove this comment
            flow *= (-1 if avg_second_track < avg_main_track else 1)
        else:
            flow *= (-1 if avg_second_track > avg_main_track else 1)
        if inter[self._direction] is not None:
            for i in range(tiles_before):
                self._next_nodes.append({inter[self._direction][0]+flow*(i):{0:direction,1:direction}})
        ###########

        self.goal_id = list(self._next_nodes[-1].keys())[0]
        self.new = True
        return Ok

    def _weak_reset_side(self):
        """
        This function takes care of ALL the processes to reset the 
        environment, if anything goes wrong, it returns false.
        This is in order to allow several retries to reset 
        the environment
        """
        filter = (self.info['x']) | ((self.info['t']))# & (self.info['track'] > 0))
        # TODO why is a track > 0?
        # Becase originally I thought having allways Left and Right was good
        if any(filter) == False:
            return False
        else:
            return self._generate_predictions_side(filter)
    
    def reset(self):
        while True:
            obs = super(Turn_side,self).reset()
            if obs is not False:
                if self._weak_reset_side():
                    break
        obs = self.step(None)[0]
        return obs

    def _remove_prediction(self, id,lane,direction):
        ###### Removing current new tile from nexts
        pass
        #if len(self._next_nodes) > 0:
            #for tiles in self._next_nodes:
                #if id in tiles:
                    #del tiles[id]

    def _check_predictions(self):
        pass

    #def step(self,action):
        #s,r,d,_ = super(Turn_side,self).step(action)
        #if self.goal_id in self._current_nodes:
            #d = True
        #return s,r,d,_

    def remove_current_tile(self, id, lane):
        if id in self._current_nodes:
            if len(self._current_nodes[id]) > 1:
                del self._current_nodes[id][lane]
            else:
                del self._current_nodes[id]


class Turn_left(Turn_side):
    def __init__(self,id='TL'):
        super(Turn_left,self).__init__(id=id)
        self._flow = 1
        self._direction = 'left'


class Turn_right(Turn_side):
    def __init__(self,id='TR'):
        super(Turn_right,self).__init__(id=id)
        self._flow = -1
        self._direction = 'right'


class Turn(Turn_side):
    def __init__(self,id='T',high_level=True,*args,**kwargs):
        super(Turn,self).__init__(id=id,high_level=high_level,*args,**kwargs)

        self._direction = 'right' if np.random.uniform() >= 0.5 else 'left'
        self._flow = -1 if self._direction == 'right' else 1

        self.actions = {}
        self.actions['left']  = Left_policy()
        self.actions['right'] = Right_policy()

    def _set_config(self, **kwargs):
        super(Turn, self)._set_config(**kwargs)
        #self.discretize_actions = 'left-right'
        self.action_space = spaces.Discrete(2)

    def reset(self):
        self._direction = 'right' if np.random.uniform() >= 0.5 else 'left'
        self._flow = -1 if self._direction == 'right' else 1
        return super(Turn,self).reset()

    def step(self,action):
        # transform action
        #if action is not None: set_trace()
        if action is None:
            state, reward, done, info = self.raw_step(None)
        else:
            action = self._transform_high_lvl_action(action)

            # execute transformed action
            state, reward, done, info = action(self,self.state)

        return state, reward, done, info
    
    def raw_step(self,action):
        # Normal step 
        return super(Turn,self).step(action)

    def _transform_high_lvl_action(self,action):
        if action == 0:
            action = self.actions['left']
        elif action == 1:
            action = self.actions['right']
        return action

    def _render_side_arrow(self):
        super(Turn,self)._render_side_arrow(self._direction, self._flow)

    def _render_additional_objects(self):
        self._render_side_arrow()


class Turn_n2n(Turn):
    def __init__(self,id='T',*args,**kwargs):
        super(Turn,self).__init__(id=id,*args,**kwargs)

        self._direction = 'right' if np.random.uniform() >= 0.5 else 'left'
        self._flow = -1 if self._direction == 'right' else 1

    def _set_config(self,**kwargs):
        super(Turn,self)._set_config(**kwargs)

    def step(self,action):
        return super(Turn,self).step(action)


class Take_center(Base):
    def __init__(self, id='TC', reward_fn=None, max_time_out=1.0,*args, **kwargs):

        def reward_fn(env):
            reward = -SOFT_NEG_REWARD
            done = False

            right_old  = env.info['count_right_delay']  > 0
            left_old  = env.info['count_left_delay']  > 0
            not_visited = env.info['visited'] == False
            track = env.info['track'] == env.track_id

            if env.goal_id in np.where(right_old|left_old)[0]:
                # If in objective
                reward += 10
                done = True
            else:
                reward,done = env.check_outside(reward,done)
                reward,done = env.check_timeout(reward,done)
                reward,done = env.check_unvisited_tiles(reward,done)
            
                # if still in the same lane and same track 
                # if still in the prediction set
                if not done and len(list(set(env.predictions_id) & set(\
                        np.where(not_visited & (right_old|left_old))[0]))) > 0:
        
                    # if in different lane than original
                    if env.lane_id == 1 and (left_old & track & not_visited).sum() > 0:
                        factor = 2
                    elif env.lane_id == 0 and (right_old & track & not_visited).sum() > 0:
                        factor = 2
                    else:
                        factor = 1 
                    reward += 1 / factor

            # Cliping reward per episode
            full_reward = reward
            reward = np.clip(
                    reward, env.min_step_reward, env.max_step_reward)

            env.info['visited'][left_old | right_old] = True
            env.info['count_right_delay'] = env.info['count_right']
            env.info['count_left_delay']  = env.info['count_left']
            
            return reward,full_reward,done

        super(Take_center,self).__init__(
                reward_fn=reward_fn,
                max_time_out=max_time_out,
                id=id,
                *args, 
                **kwargs,
                )
        self.predictions_id = []
        self._reward_fn_center = reward_fn

    def reset(self):
        self.predictions_id = []
        while True: 
            obs = super(Take_center,self).reset()
            if obs is not False:
                if self._weak_reset_center():
                    break
        obs = self.step(None)[0]
        return obs

    def _generate_predictions_center(self,filter):
        Ok = True

        # Chose start randomly before or after intersection
        tiles_before = -1 if np.random.uniform() < 0.5 else 1 
        tiles_before *= 8

        # original point
        idx_org = np.random.choice(np.where(filter)[0])
        idx_relative = idx_org - (self.info['track'] < self.info[idx_org]['track']).sum()

        intersection = self.understand_intersection(idx_org,-np.sign(tiles_before))
        if intersection['straight'] == None: return False

        # Get start
        track = self.track[self.info['track'] == self.info[idx_org]['track']]
        idx_general = (idx_relative - tiles_before)%len(track) + (self.info['track'] < self.info[idx_org]['track']).sum()
        start = self.track[idx_general][0]
        if -tiles_before > 0: start[1] += np.pi # beta
        self.start_id = idx_general
        self.track_id = self.info[idx_general]['track']

        # Create predictions before intersection
        predictions_before = list(range(abs(tiles_before)))
        predictions_before = [ idx_relative-tiles_before+np.sign(tiles_before)*id for id in predictions_before ] 
        predictions_before = [ id % len(track) \
                for id in predictions_before ]
        predictions_before = [ id + (self.info['track'] < self.track_id).sum() \
                for id in predictions_before ]

        # Get goal
        straight = intersection['straight'][0] 
        track = self.track[self.info['track'] == self.info[straight]['track']]
        idx_relative = straight - (self.info['track'] < self.info[straight]['track']).sum()
        idx_general = (idx_relative + tiles_before)%len(track) + (self.info['track'] < self.info[straight]['track']).sum()
        self.goal_id  = idx_general

        # Create predictions after intersection
        predictions_after = list(range(abs(tiles_before)))
        predictions_after = [ idx_relative+np.sign(tiles_before)*id for id in predictions_after ] 
        predictions_after = [ id % len(track) \
                for id in predictions_after ]
        predictions_after = [ id + (self.info['track'] < self.info[straight]['track']).sum() \
                for id in predictions_after ]

        # Get a random position in a lane 
        lane = 0 if np.random.uniform() < 0.5 else 1
        delta = np.random.uniform(low=0.0)
        _,beta,x,y = start
        x = x + (-(-1)**lane)*np.cos(beta)*delta*(TRACK_WIDTH)
        y = y + (-(-1)**lane)*np.sin(beta)*delta*(TRACK_WIDTH)
        angle_noise = np.random.uniform(-1,1)*np.pi/8
        beta += angle_noise # orientation with noise
        self.place_agent([beta,x,y])
        speed = np.random.uniform(0,100)
        self.set_speed(speed)

        # Save the current lane and track
        self.lane_id = lane if tiles_before > 0 else 1-lane 

        self.predictions_id = predictions_before + predictions_after

        return Ok

    def _weak_reset_center(self):
        # Finding 'x' intersection
        filter = self.info['x'] == True
        if filter.sum() == 0:
            return False
        return self._generate_predictions_center(filter)

    def update_contact_with_track(self):
        self.update_contact_with_track_center()

    def update_contact_with_track_center(self):
        # Only updating it if still in prediction

        not_visited = self.info['visited'] == False
        right = self.info['count_right'] > 0
        left  = self.info['count_left']  > 0

        # Intersection of the prediction and the set that you are currently in
        if len(list( set(self.predictions_id) & set(\
                np.where(right|left)[0]))) > 0:
            #super(Take_center,self).update_contact_with_track()
            self.last_touch_with_track = self.t

    def _render_additional_objects(self):
        self._render_center_arrow()


class X(Turn,Take_center):
    def __init__(self, 
            left_count=0,
            right_count=0,
            center_count=0,
            total_tracks_generated=0,
            is_current_type_side=None, 
            reward_fn=None, 
            max_step_reward=10,
            id='X',
            high_level=True,
            *args, **kwargs):

        def reward_fn(env):
            if env.is_current_type_side:
                return env._reward_fn_side(env)
            else:
                return env._reward_fn_center(env)

        super(X,self).__init__(
                id=id,
                max_step_reward=max_step_reward,
                high_level=high_level,
                *args,**kwargs)
        self.is_current_type_side = is_current_type_side
        self.reward_fn = reward_fn
        self.reward_fn_X = reward_fn

        self.actions = {}
        self.actions['turn'] = Turn_policy()
        self.actions['take_center'] = Take_center_policy()

        self.stats['left_count'] = left_count
        self.stats['right_count'] = right_count
        self.stats['center_count'] = center_count
        self.stats['total_tracks_generated'] = total_tracks_generated

        self.tracks_df = self.tracks_df[self.tracks_df['x'] == True]

    def _set_config(self, **kwargs):
        super(X, self)._set_config(**kwargs)
        self.action_space = spaces.Discrete(2)

    def reset(self):
        while True:
            obs = super(Take_center,self).reset() # Just calls base.reset
            self.stats['total_tracks_generated'] += 1
            #filter = (self.info['x']) | ((self.info['t']) & (self.info['track'] >0))
            filter = (self.info['x'])# & (self.info['track'] == 0))
            if any(filter):
                # chose randomly between Take_center and Turn 
                # 2/3 because sides are 50/50 left and right
                self.is_current_type_side = True if np.random.uniform() < 2/3 else False
                if self.is_current_type_side:
                    self._direction = 'right' if np.random.uniform() >= 0.5 else 'left'
                    self._flow = -1 if self._direction == 'right' else 1

                    ok = self._generate_predictions_side(filter)
                else:
                    ok = self._generate_predictions_center(filter)

                if ok:
                    if self.is_current_type_side:
                        if self._direction == 'right':
                            self.stats['right_count'] += 1
                        else:
                            self.stats['left_count'] += 1
                    else:
                        self.stats['center_count'] += 1
                    break
            else:
                continue
        obs = self.step(None)[0]
        return obs

    def step(self,action):
        # return super(Turn,self).step(action) # To n2n
        # transform action
        #if action is not None: set_trace()
        if action is None:
            state, reward, done, info = self.raw_step(None)
        else:
            action = self._transform_high_lvl_action(action)

            # execute transformed action
            state, reward, done, info = action(self,self.state)

        return state, reward, done, info
    
    def _transform_high_lvl_action(self,action):
        if action == 0:
            action = self.actions['turn']
        elif action == 1:
            action = self.actions['take_center']
        return action

    def weak_reset(self):
        raise NotImplementedError

    def update_contact_with_track(self):
        if self.is_current_type_side:
            self.update_contact_with_track_side()
        else:
            self.update_contact_with_track_center()

    def _render_additional_objects(self):
        if self.is_current_type_side:
            self._render_side_arrow()
        else:
            self._render_center_arrow()


class X_n2n(X):
    def __init__(self, 
            left_count=0,
            right_count=0,
            center_count=0,
            total_tracks_generated=0,
            is_current_type_side=None, 
            reward_fn=None, 
            max_step_reward=10,
            high_level=False,
            *args, **kwargs):

        def reward_fn(env):
            if env.is_current_type_side:
                reward,full_reward,done = env._reward_fn_side(env)
            else:
                reward,full_reward,done = env._reward_fn_center(env)
            return reward,full_reward,done

        super(X,self).__init__(
                id=id,
                max_step_reward=max_step_reward,
                high_level=False,
                *args,**kwargs)
        self.is_current_type_side = is_current_type_side
        self.reward_fn = reward_fn
        self.reward_fn_X = reward_fn

        self.stats['left_count'] = left_count
        self.stats['right_count'] = right_count
        self.stats['center_count'] = center_count
        self.stats['total_tracks_generated'] = total_tracks_generated

        self.tracks_df = self.tracks_df[self.tracks_df['x'] == True]

    def step(self,action):
        return self.raw_step(action) # To n2n


class Keep_lane(Base):
    def __init__(self, id='KL', allow_outside=False, *args,**kwargs):
        def reward_fn(env):
            # Ignore the obstacles
            env._ignore_obstacles()

            # if touching other track reward = 0
            in_other_lane = 1
            if env.keeping_left:
                if env.info['count_right_delay'].sum() > 0:
                    in_other_lane=0
            else:
                if env.info['count_left_delay'].sum() > 0:
                    in_other_lane=0

            # if close to an intersection done=True
            _done = env._check_if_close_to_intersection()

            # calculate normal reward
            reward,full_reward,done = default_reward_callback(env)

            # For child classes
            reward,full_reward,done = \
                    self._check_early_termination_change_lane(
                            reward,full_reward,done)

            done = done if not _done else _done

            if reward > 0:
                reward *= in_other_lane
                full_reward *= in_other_lane
            return reward,full_reward,done

        super(Keep_lane, self).__init__(*args, **kwargs, id=id, allow_outside=allow_outside)
        self.reward_fn = reward_fn

    def _check_early_termination_change_lane(self,reward,full_reward,done):
        return reward,full_reward,done

    def _set_side(self):
        self.keeping_left = True if np.random.uniform() >= 0.5 else False

    def reset(self):
        self._set_side()
        while True:
            obs = super(Keep_lane,self).reset()
            if obs is not False:
                if self._weak_reset_keep_lane():
                    break

        return self.step(None)[0]

    def _weak_reset_keep_lane(self):
        # Place the agent randomly in a good position
        while True:
            tile_id = np.random.choice(list(range(len(self.track))))
            if not self._is_close_to_intersection(tile_id,8):
                break

        _,beta,x,y = self._get_position_inside_lane(
                tile_id,x_pos=self.keeping_left,discrete=True)
        self.place_agent([beta,x,y])

        return True


class Y(Turn):
    pass


class NWOO_n2n(Base):
    def __init__(self, id='NWOO', ignore_obstacles=True, allow_outside=False, *args, **kwargs):
        def reward_fn(env):
            if env._ignore_obstacles is False:
                env._ignore_obstacles()
            reward,full_reward,done = default_reward_callback(env)

            current_nodes = list(env._current_nodes.keys())
            if env._objective in current_nodes:
                # Changing from close to not close
                reward = reward + 100
                full_reward = full_reward + 100

            reward,full_reward,done = env._check_early_termination_NWO(reward,full_reward,done)
            reward,full_reward,done = env._check_if_in_objective(reward,full_reward,done)
            return reward,full_reward,done

        Base.__init__(
                self, 
                id=id, 
                allow_outside=allow_outside, 
                reward_fn=reward_fn, 
                *args, **kwargs)

        self._ignore_obstacles = ignore_obstacles
        self._close_to_intersection_state = False
        self._reward_fn_NWOO_n2n = reward_fn

    def _check_early_termination_NWO(self,reward,full_reward,done):
        # for NWO
        return reward,full_reward,done

    def _check_if_in_objective(self,reward,full_reward,done):
        current_nodes = list(self._current_nodes.keys())
        if len(set(current_nodes).intersection(
                self._neg_objectives + [self._objective])) > 0:
            self._close_to_intersection_state = False
            self._directional_state = None
            self._objective = None
            self._neg_objectives = []

            # Clean visited tiles
            self.info['visited'] = False
        return reward,full_reward,done

    def reset(self):
        self._clean_NWOO_n2n_vars()
        return super(NWOO_n2n,self).reset()

    def _clean_NWOO_n2n_vars(self):
        self._directional_state = None
        self._close_to_intersection_state = False
        self._objective = None
        self._neg_objectives = []

    def _check_and_set_objectives(self):
        # if close to an intersection, chose a direction and set a 
        # goal, all the other points will be a big negative reward

        dirs = set(val \
                for elm in self._current_nodes.values() \
                for val in elm.values())

        # Set longitudinal direction
        if len(dirs) == 1:
            direction = dirs.pop()
            self._long_dir = direction
        else:
            self._long_dir = None

        if self._long_dir is not None:
            current_tile = list(self._current_nodes.keys())[0]
            intersection_tiles = self.get_close_intersections(
                    current_tile,spaces=8,direction=direction)

            if len(intersection_tiles) > 0:
                if not self._close_to_intersection_state:
                    # changing state to close
                    self._close_to_intersection_state = True

                    # Choose positive and negative goals
                    intersection_tiles = intersection_tiles.intersection(
                            np.where(self.info['track']==self.info[current_tile]['track'])[0])
                    intersection_tile = intersection_tiles.pop()
                    intersection_dict = self.understand_intersection(
                            intersection_tile,direction)

                    options = self._get_options_for_directional(intersection_dict)

                    self._directional_state = np.random.choice(options)

                    # set positive and negative objectives
                    for directional, val in intersection_dict.items():
                        if val is not None:
                            tmp_id, tmp_flow = val

                            track_id = self.info[tmp_id]['track']
                            track_len = len(self.tracks[track_id])
                            comp_track_len = sum(self.info['track'] < track_id)
                            objective = ((tmp_id - comp_track_len + tmp_flow*8) \
                                    % track_len) + comp_track_len
                            if directional == self._directional_state:
                                self._objective = objective
                            else:
                                self._neg_objectives.append(objective)

    def _get_options_for_directional(self,intersection):
        return [key for key,val in intersection.items() if val is not None]

    def step(self,action):
        self._check_and_set_objectives()
        return Base.step(self,action)

    def _render_additional_objects(self):
        if self._close_to_intersection_state == True:
            # Render the appropiate arrow
            if self._directional_state == 'straight':
                self._render_center_arrow()
            elif self._directional_state == 'left':
                self._render_side_arrow('left',self._long_dir)
            elif self._directional_state == 'right':
                self._render_side_arrow('right',self._long_dir)


class NWOO(High_level_env_extension,NWOO_n2n):
    """
    actions are 1: keep_lane, 2: x, 3: y
    """
    def __init__(self,id="NWOO",high_level=True,*args,**kwargs):
        self.actions = []
        self.actions.append(Keep_lane_policy())
        self.actions.append(X_policy())
        self.actions.append(Y_policy())

        super(NWOO,self).__init__(id=id,high_level=high_level,*args,**kwargs)

    def step(self,action):
        if action is not None:
            self._check_and_set_objectives()

        state, reward, done, info = super(NWOO,self).step(action) 

        return state, reward, done, info


class NWO_n2n(NWOO_n2n):
    def __init__(self,ignore_obstacles=True,*args,**kwargs):
        super(NWO_n2n,self).__init__(ignore_obstacles=ignore_obstacles,*args,**kwargs)

    def _check_early_termination_NWO(self,reward,full_reward,done):
        # if close to an interseciton return done=True
        if len(list(self._current_nodes.keys())) > 0 and self._long_dir is not None:
            current_node = list(self._current_nodes.keys())[0]
            if self._is_close_to_intersection(current_node,10,self._long_dir):
                done = True
        return reward,full_reward,done


class NWO(High_level_env_extension,NWO_n2n):
    def __init__(self,id='NWO',*args,**kwargs):
        self.actions = []
        self.actions.append(Keep_lane_policy())
        self.actions.append(Change_lane_policy())

        super(NWO,self).__init__(id=id,*args,**kwargs)


class Turn_v2_n2n(NWOO_n2n):
    def _get_options_for_directional(self,intersection):
        if not None in intersection.values():
            intersection = deepcopy(intersection)
            intersection['straight'] = None
        options = super(Turn_v2_n2n,self)._get_options_for_directional(intersection)
        return options

    def _choice_random_track_from_file(self):
        # x has priority when there are x in track
        # so to balance things out, here t has some bias
        if np.random.uniform() >= 0.7: 
            idx = np.random.choice(self.tracks_df[self.tracks_df['x']].index)
        else:
            idx = np.random.choice(self.tracks_df[self.tracks_df['t']].index)
        return idx

    def _check_if_in_objective(self,reward,full_reward,done):
        current_nodes = list(self._current_nodes.keys())
        done_ = False
        if len(set(current_nodes).intersection(
                self._neg_objectives + [self._objective])) > 0:
            done_ = True
        super(Turn_v2_n2n,self)._check_if_in_objective(reward,full_reward,done)

        done = True if done_ else done
        return reward,full_reward,done

    def reset(self):
        obs = super(Turn_v2_n2n,self).reset()

        beta,x,y = self.get_position_near_junction('xt',8)
        angle_noise = np.random.uniform(-1,1)*np.pi/12
        beta += angle_noise

        self.place_agent([beta,x,y])

        speed = 0
        if np.random.uniform() > 0.2:
            speed = np.random.uniform(0,100)
        self.set_speed(speed)

        for _ in range(self.frames_per_state):
            obs = self.step(None)[0]

        self._clean_NWOO_n2n_vars()
        return obs


class Turn_side_v2(Turn_v2_n2n):
    """
    side_to_turn is whether 'left' or 'right'
    """
    def __init__(self,side_to_turn,*args,**kwargs):
        super(Turn_side_v2,self).__init__(*args,**kwargs)
        self._side_to_turn = side_to_turn

    def _get_options_for_directional(self,intersection):
        options = super(Turn_side_v2,self)._get_options_for_directional(intersection)
        if self._side_to_turn in options:
            options = [self._side_to_turn]
        else:
            options = ['straight']
        return options


class Turn_right_v2(Turn_side_v2):
    def __init__(self,*args,**kwargs):
        super(Turn_right_v2,self).__init__(side_to_turn='right',*args,**kwargs)


class Turn_left_v2(Turn_side_v2):
    def __init__(self,*args,**kwargs):
        super(Turn_left_v2,self).__init__(side_to_turn='left',*args,**kwargs)


class Change_lane_n2n(Keep_lane):
    def __init__(self, id='CLane', *args,**kwargs):
        super(Change_lane_n2n,self).__init__(id=id,*args,**kwargs)

    def _get_position_inside_lane(self,idx,x_pos,border=True,direction=1,discrete=False):
        x_pos = 1 - x_pos
        return super(Change_lane_n2n,self)._get_position_inside_lane(
                idx,
                x_pos,
                border=border,
                direction=direction,
                discrete=discrete)

    def _check_early_termination_change_lane(self,reward,full_reward,done):
        if self.full_reward >= 10:
            done = True
        return reward,full_reward,done

    def reset(self):
        obs = super(Change_lane_n2n,self).reset()
        speed = np.random.uniform(0,150)
        self.set_speed(speed)

        for _ in range(self.frames_per_state):
            obs = self.step(None)[0]
        return obs


class Change_lane(High_level_env_extension,Change_lane_n2n):
    def __init__(self,*args,**kwargs):
        self.actions = []
        self.actions.append(Change_to_left_policy())
        self.actions.append(Change_to_right_policy())

        super(Change_lane,self).__init__(*args,**kwargs)


    def _check_early_termination_change_lane(self,reward,full_reward,done):
        if self._steps_taken > 2:
            done = True

        return reward,full_reward,done

    def reset(self):
        self._steps_taken = 0
        return super(Change_lane,self).reset()

    def step(self,action):
        if action is not None: self._steps_taken += 1
        return super(Change_lane,self).step(action)


class Change_to_left(Change_lane_n2n):
    def __init__(self, id='CLeft', *args,**kwargs):
        super(Change_lane_n2n,self).__init__(id=id,*args,**kwargs)

    def _set_side(self):
        self.keeping_left = True


class Change_to_right(Change_lane_n2n):
    def __init__(self, id='CRight', *args,**kwargs):
        super(Change_lane_n2n,self).__init__(id=id,*args,**kwargs)

    def _set_side(self):
        self.keeping_left = False


def play_high_level(env):
    """
    Extension of play function in car_racing for high level policies

    env:        CarRacing env
    """
    from pyglet.window import key
    a = np.array([0])
    def key_press(k, mod):
        if k==key._1: a[0] = 0
        if k==key._2: a[0] = 1
        if k==key._3: a[0] = 2
        if k==key._4: a[0] = 3
    def key_release(k, mod):
        a[0] = -1
        if k==key.D:     set_trace()
        if k==key.R:     env.reset()
        if k==key.Z:     env.change_zoom()
        if k==key.G:     env.switch_intersection_groups()
        if k==key.I:     env.switch_intersection_points()
        if k==key.X:     env.switch_xt_intersections()
        if k==key.E:     env.switch_end_of_track()
        if k==key.S:     env.switch_start_of_track()
        if k==key.T:     env.screenshot('./')
        if k==key.Q:     sys.exit()

    env.render()
    record_video = False
    if record_video:
        env.monitor.start('/tmp/video-test', force=True)
    env.key_press_fn = key_press
    env.key_release_fn = key_release
    while True:
        env.reset()
        total_reward = 0.0
        steps = 0
        restart = False

        while True:
            a_tmp = a[0]
            if a_tmp != -1:
                s, r, done, info = env.step(a_tmp)
                total_reward += r
                if steps % 200 == 0 or done:
                    #print("\naction " + str(["{:+0.2f}".format(x) for x in a]))
                    print("step {} total_reward {:+0.2f}".format(steps, total_reward))
                    steps += 1
                    print("step {} total_reward {:+0.2f}".format(steps, total_reward))
                steps += 1
                if done or restart: break
            if not record_video: # Faster, but you can as well call env.render() every time to play full window.
                env.render()

    env.close()

if __name__=='__main__':
    args = get_env_args()
    env = getattr(environments, args.env)()
    if env.high_level: 
        print("HIGH LEVEL")
        env.auto_render = True
        play_high_level(env)
    else:
        play(env)

