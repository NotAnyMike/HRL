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
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt


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
        self.screen = pygame.display.set_mode((800, 600))
        self.clock = pygame.time.Clock()

        my_dpi = 100
        self.fig = Figure(figsize=(4,4),dpi=my_dpi)
        self.canvas = FigureCanvas(self.fig)
        self.ax = self.fig.gca()

        self.fig2 = Figure(figsize=(4,4),dpi=my_dpi)
        self.canvas2 = FigureCanvas(self.fig2)
        self.ax2 = self.fig2.gca()
        
        self.data = [1,2,7,3,7,3,8]

    def plot(self):
        self.data.append(np.random.uniform()*10)
        if len(self.data) > 20:
            self.data = self.data[-20:]

        self.ax.clear()
        self.ax2.clear()
        color = (255, 100, 0)
        self.screen.fill((255, 255, 255))
        #self.clock.tick(30)


        self.ax2.plot(self.data,c='b')
        
        # Build a dataframe with your connections
        connections = [
            ['Nav','NWOO'],
            ['Nav','NWO'],
            ['Nav','R'],
            ['NWOO','X'],
            ['NWOO','Y'],
            ['NWOO','KL'],
            ['NWO','KL'],
            ['NWO','CLane'],
            ['R','D'],
            ['R','De'],
            ['X','TC'],
            ['X','T'],
            ['Y','T'],
            ['CLane','CL'],
            ['CLane','CR'],
            ['T','TL'],
            ['T','TR'],
        ]
        from_nodes = list(zip(*connections))[0]
        to_nodes =   list(zip(*connections))[1]
        value = ['#333333']*len(from_nodes)

        pos = {'Nav': (50, 40),
               'NWOO': (40,30),
               'NWO':(50,30),
               'R': (60,30),
               'X':(35,20),
               'Y':(40,20),
               'KL':(45,20),
               'CLane':(50,20),
               'D':(57.5,20),
               'De':(62.5,20),
               'TC':(32.5,10),
               'T':(37.5,10),
               'CL':(47,10),
               'CR':(53,10),
               'TL':(35,0),
               'TR':(40,0)}


        df = pd.DataFrame({ 'from':from_nodes, 'to': to_nodes, 'value':value})

        # Build your graph
        G=nx.from_pandas_edgelist(df, 'from', 'to', create_using=nx.Graph() )

        # Custom the nodes:
        nx.draw(G, pos, ax=self.ax, 
                with_labels=True, 
                node_color='#dee0e2', 
                node_size=1000, 
                font_size=8, 
                edge_color=df['value'], 
                width=2.0, 
                edge_cmap=plt.cm.Blues, 
                node_shape='s')

        self.fig.tight_layout()
        self.fig2.tight_layout()
        self.canvas.draw()       # draw the canvas, cache the renderer
        self.canvas2.draw()

        image = self.canvas.tostring_rgb()
        image2 = self.canvas2.tostring_rgb()

        size = self.canvas.get_width_height()
        size2 = self.canvas2.get_width_height()

        surf = pygame.image.fromstring(image, size, "RGB")
        surf2 = pygame.image.fromstring(image2, size2, "RGB")
        self.screen.blit(surf, (0,0))
        self.screen.blit(surf2, (400,0))
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
