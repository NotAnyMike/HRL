import numpy as np
from pdb import set_trace

from gym.envs.box2d import CarRacing
from gym.envs.box2d.car_racing import play, TILE_NAME, default_reward_callback

class Base(CarRacing):
    def __init__(self, reward_fn=default_reward_callback,max_episode_reward=1):
        super(Base,self).__init__(
                allow_reverse=False, 
                grayscale=1,
                show_info_panel=False,
                verbose=0,
                discretize_actions="hard",
                num_tracks=2,
                num_lanes=1,
                num_lanes_changes=4,
                max_time_out=2,
                frames_per_state=4,
                max_episode_reward=max_episode_reward,
                reward_fn=reward_fn,
                )

class Turn_left(Base):
    def __init__(self):
        def reward_fn(tile,obj,begin,local_vars,global_vars):
            # Substracting value of obstacle
            self = local_vars['self']
            predictions_id = [id for l in self.env._next_nodes for id in l.keys() ]
            if begin:

                if tile.typename == TILE_NAME:
                    self.env.add_current_tile(tile.id, tile.lane)
                obj.tiles.add(tile)

                # Checking if it was visited before
                #if len(self.env._next_nodes) > 0 and tile.id in self.env._next_nodes[0]:
                if tile.id in predictions_id and not tile.road_visited:
                    tile.road_visited = True
                    if self.env.goal_id == tile.id:
                        reward_episode = 10
                        d = True
                    #elif action is not None and len(self._current_nodes) == 0:
                        ## in case it is outside the track --> not working
                        #d = True
                        #r = -1000
                    else:
                        reward_episode = 1

                    # Cliping reward per episode
                    reward_episode = np.clip(
                            reward_episode, self.env.min_episode_reward, self.env.max_episode_reward)
                    self.env.reward += reward_episode
                    self.env.tile_visited_count += 1
            else:
                obj.tiles.remove(tile)
                if self.env.new == False and not local_vars['contact'].touching:
                    self.env.remove_current_tile(tile.id, tile.lane)
                # Registering last contact with track
                if tile.id in predictions_id:
                    self.env.last_touch_with_track = self.env.t

        super(Turn_left,self).__init__(reward_fn=reward_fn,max_episode_reward=10)
        self.goal_id = None
        self.new = True

    def _weak_reset(self):
        """
        This function takes care of ALL the processes to reset the 
        environment, if anything goes wrong, it returns false.
        This is in order to allow several retries to reset 
        the environment
        """
        to_return = super(Turn_left,self).reset()
        tiles_before = 8
        filter = (self.info['x']) | ((self.info['t']) & (self.info['track'] >0))
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

        # The direction of left lane
        flow = -1 if avg_second_track > avg_main_track else 1
        if current_track == 0: flow *= -1
        if inter['left'] is not None:
            for i in range(tiles_before):
                self._next_nodes.append({inter['left']+flow*(i):{0:direction,1:direction}})
        ###########

        self.goal_id = list(self._next_nodes[-1].keys())[0]
        self.new = True
        return to_return
    
    def reset(self):
        while True:
            to_return = self._weak_reset()
            if to_return is not False:
                break
        return to_return

    def _remove_prediction(self, id,lane,direction):
        ###### Removing current new tile from nexts
        pass
        #if len(self._next_nodes) > 0:
            #for tiles in self._next_nodes:
                #if id in tiles:
                    #del tiles[id]

    def _check_predictions(self):
        pass

    def step(self,action):
        # TODO check if it is time to reset 
        if action is not None:
            self.new = False
        s,r,d,_ = super(Turn_left,self).step(action)
        if self.goal_id in self._current_nodes:
            d = True
        return s,r,d,_

    def remove_current_tile(self, id, lane):
        if id in self._current_nodes:
            if len(self._current_nodes[id]) > 1:
                del self._current_nodes[id][lane]
            else:
                del self._current_nodes[id]

if __name__=='__main__':
    env = Base()
    play(env)
