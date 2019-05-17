import matplotlib
matplotlib.use("Agg")

import matplotlib.backends.backend_agg as agg
import pygame
import time

import multiprocessing as mp
import cloudpickle
import pickle

from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure

import numpy as np

def _worker(wrapper):
    plotter = wrapper.var()
    keep_running = True
    while keep_running == True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                keep_running = False
        plotter.plot()

class Plotter():
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((400, 400))
        self.clock = pygame.time.Clock()

        my_dpi = 100
        self.fig = Figure(figsize=(3,3),dpi=my_dpi)
        self.canvas = FigureCanvas(self.fig)
        self.ax = self.fig.gca()
        
        self.data = [1,2,7,3,7,3,8]

    def plot(self):
        self.data.append(np.random.uniform()*10)
        if len(self.data) > 20:
            self.data = self.data[-20:]
        print(self.data)

        self.ax.clear()
        color = (255, 100, 0)
        self.screen.fill((255, 255, 255))
        #self.clock.tick(30)

        self.ax.plot(self.data,c='b')
        self.fig.tight_layout()

        self.canvas.draw()       # draw the canvas, cache the renderer

        image = self.canvas.tostring_rgb()

        size = self.canvas.get_width_height()

        surf = pygame.image.fromstring(image, size, "RGB")
        self.screen.blit(surf, (0,0))
        pygame.display.flip() 
        #pygame.display.update()

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
    process.join()
    print("exiting ...")
