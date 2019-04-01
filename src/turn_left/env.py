from gym.envs.box2d import CarRacing
from gym.envs.box2d.car_racing import play

class CarRacing_turn_left(CarRacing):
    def __init__(self):
        super(CarRacing_turn_left,self).__init__(
                allow_reverse=False, 
                grayscale=0,
                show_info_panel=1,
                discretize_actions=None,
                num_tracks=2,
                num_lanes=2,
                num_lanes_changes=4,
                max_time_out=0,
                frames_per_state=4)
    
    def reset(self):
        super(CarRacing_turn_left,self).reset()
        pos = self.get_position_near_junction('xt',8)
        self.place_agent(pos)

    def step(self,action):
        # TODO check if it is time to reset 
        return super(CarRacing_turn_left,self).step(action)

if __name__=='__main__':
    env = CarRacing_turn_left()
    play(env)
