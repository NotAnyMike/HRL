import matplotlib
matplotlib.use("Agg")

import matplotlib.backends.backend_agg as agg
import pygame
import time

import multiprocessing as mp
import cloudpickle
import pickle

def _worker(wrapper):
    plotter = wrapper.var()
    plotter.plot()

class Plotter():
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((400, 400))
        self.clock = pygame.time.Clock()

    def plot(self):
        for i in range(500):
            pygame.event.pump()
            color = (255, 100, 0)
            self.screen.fill((0, 0, 0))
            pygame.draw.rect(self.screen, color, pygame.Rect(int(i*400/1000), int(i*400/1000), 60, 60))
            #pygame.display.flip()
            pygame.display.update()
            #self.clock.tick(40)

class PickleWrapper():
    def __init__(self,var):
        self.var = var
        
    def __getstate__(self):
        return cloudpickle.dumps(self.var)

    def __setstate__(self, obs):
        self.var = pickle.loads(obs)

if __name__=='__main__':
    to_pickle = lambda: Plotter()
    ctx = mp.get_context('spawn')
    process = ctx.Process(target=_worker,args=(PickleWrapper(to_pickle),),daemon=True)
    process.start()
    i = 0
    while 1:
        print(i)
        i += 1
