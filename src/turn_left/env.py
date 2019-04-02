import numpy as np

from gym.envs.box2d import CarRacing
from gym.envs.box2d.car_racing import play

class CarRacing_turn(CarRacing):
    def __init__(self):
        super(CarRacing_turn,self).__init__(
                allow_reverse=False, 
                grayscale=0,
                show_info_panel=1,
                discretize_actions=None,
                num_tracks=2,
                num_lanes=1,
                num_lanes_changes=4,
                max_time_out=0,
                frames_per_state=4)
    
    def reset(self):
        super(CarRacing_turn,self).reset()
        # TODO take only t if we are in the second track
        tiles_before = 8
        filter = (env.info['x']) | ((self.info['t']) & (self.info['track'] >0))
        idx = np.random.choice(np.where(filter)[0])

        idx_relative = idx - (self.info['track'] < self.info[idx]['track']).sum()
        if env.info[idx]['end']:
            direction = -1
        elif env.info[idx]['start']:
            direction = 1
        else:
            direction = 1 if np.random.random() < 0.5 else -1
        idx_relative_new = (idx_relative + direction*tiles_before) % len(env.tracks[self.info[idx]['track']])

        idx_new = idx - idx_relative + idx_relative_new
        _,beta,x,y = env._get_rnd_position_inside_lane(idx_new,direction=-1*direction)            

        angle_noise = np.random.uniform(-1,1)*np.pi/12
        beta += angle_noise

        self.place_agent([beta,x,y])
        speed = np.random.uniform(0,100)
        self.set_speed(0)#speed)

        self._next_nodes = []
        for i in range(tiles_before):
            idx_tmp = (idx_relative + direction*i) % len(env.tracks[self.info[idx]['track']])
            idx_tmp = idx - idx_relative + idx_tmp
            self._next_nodes.append({idx_tmp: {0:-direction,1:-direction}})

    def _remove_prediction(self, id,lane,direction):
        pass

    def _check_predictions(self):
        pass

    def step(self,action):
        # TODO check if it is time to reset 
        return super(CarRacing_turn,self).step(action)

if __name__=='__main__':
    env = CarRacing_turn()
    play(env)
