import cloudpickle
import multiprocessing as mp
import os
import pickle
import time
from copy import copy

import matplotlib
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
import networkx as nx
import numpy as np
import pandas as pd

os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"
import pygame

matplotlib.use("Agg")

def worker(Plotter,connection):
    plotter = Plotter.var()
    plotter.plot()
    keep_running = True
    while keep_running == True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                keep_running = False

        if connection.poll():
            cmd,val = connection.recv()
        else:
            continue

        getattr(plotter,cmd)(*val[0],**val[1])
        plotter.plot() # after recv to avoid plot without new info

class Plotter():
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((800, 400))
        self.clock = pygame.time.Clock()

        my_dpi = 100
        self.fig = Figure(figsize=(4,4),dpi=my_dpi)
        self.canvas = FigureCanvas(self.fig)
        self.ax = self.fig.gca()
        self.ax.set_xlim(-10,70)
        self.ax.set_ylim(-50,90)

        self.fig2 = Figure(figsize=(4,4),dpi=my_dpi)
        self.canvas2 = FigureCanvas(self.fig2)
        self.ax2 = self.fig2.gca()

        self.init_data()
        
        self.data = [1,2,7,3,7,3,8]
        self.active_nodes = set()

    def init_data(self):
        self.connections = [['Nav','NWOO'],['Nav','NWO'],['Nav','R'],
            ['NWOO','X'],['NWOO','Y'],['NWOO','KL'],['NWO','KL'],
            ['NWO','CLane'],['R','D'],['R','De'],['X','TC'],
            ['X','T'],['Y','T'],['CLane','CL'],['CLane','CR'],
            ['T','TL'],['T','TR'],
        ]
        self.from_nodes = list(zip(*self.connections))[0]
        self.to_nodes =   list(zip(*self.connections))[1]
        self.value = ['#333333']*len(self.from_nodes)

        self.pos = {'Nav': (50, 40),'NWOO': (40,30),'NWO':(50,30),
                'R': (60,30),'X':(35,20),'Y':(40,20),'KL':(45,20),
                'CLane':(50,20),'D':(57.5,20),'De':(62.5,20),
                'TC':(32.5,10),'T':(37.5,10),'CL':(47,10),
                'CR':(53,10),'TL':(35,0),'TR':(40,0)}

        self.color = ['#dee0e2']*len(self.pos)

        self.df = pd.DataFrame({ 'from':self.from_nodes, 'to': self.to_nodes, 'value':self.value})

    def add_active_policies(self,policy_names):
        for name in policy_names:
            self.add_active_policy(name)

    def add_active_policy(self,policy_name):
        """
        Policy name is the ID name of the policies, e.g. Nav, TR or NWOO
        """
        id = list(self.pos.keys()).index(policy_name)
        self.active_nodes.add(id)

    def remove_active_policy(self,policy_name):
        """
        Policy name is the ID name of the policies, e.g. Nav, TR or NWOO
        """
        id = list(self.pos.keys()).index(policy_name)
        self.active_nodes.remove(id)

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

        # Build your graph
        G=nx.from_pandas_edgelist(self.df, 'from', 'to', create_using=nx.Graph() )

        # TODO remove this
        #self.active_nodes = set(np.random.choice(range(len(self.pos)),size=3,replace=False))

        color = copy(self.color)
        for i in self.active_nodes:
            color[i] = '#e86d6d'

        # Custom the nodes:
        nx.draw(G, self.pos, ax=self.ax, 
                with_labels=True, 
                node_color=color, 
                node_size=1000, 
                font_size=8, 
                edge_color=self.df['value'], 
                width=2.0, 
                node_shape='s')

        self.fig.tight_layout()
        self.fig2.tight_layout()

        # draw the canvas, cache the renderer
        self.canvas.draw()       
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
    to_pickle = PickleWrapper(lambda: Plotter())
    parent_conn,child_conn = mp.Pipe()
    args = (to_pickle, child_conn,)
    ctx = mp.get_context('spawn')
    process = ctx.Process(target=worker,args=args,daemon=True)
    process.start()
    child_conn.close()

    parent_conn.send(('add_active_policy',[['TR'],{}]))

    time.sleep(5)

    parent_conn.send(('remove_active_policy',[['TR'],{}]))
    parent_conn.send(('add_active_policy',[['TL'],{}]))

    print("sent")
    process.join()
    print("exiting ...")
