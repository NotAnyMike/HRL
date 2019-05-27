import multiprocessing as mp
import os
import pickle

import pandas as pd
import numpy as np
import tqdm
from PIL import Image

from hrl.envs.env import Base

def worker(connection):
    env = Base(load_tracks_from=None)
    env.change_zoom()
    while True:
        env.reset()
        obs = env.render('rgb_array')
        connection.send((obs,env.track,env.tracks,env.info))
        

def generate_tracks(n_tracks,n_cpu):
    if not os.path.isdir('tracks'):
        os.makedirs('tracks')
        df = pd.DataFrame(columns=['x','t','obstacles'])
    else:
        df = pd.read_csv("tracks/list.csv",index_col=0)

    print("\n#########")
    print("Generating",n_tracks,"tracks in ", n_cpu, "cpus")

    connections = []
    processes   = []
    ctx = mp.get_context('spawn')

    # Creating processes
    for _ in range(n_cpu):
        parent_conn, child_conn = ctx.Pipe()
        connections.append(parent_conn)
        process = ctx.Process(target=worker,args=(child_conn,),daemon=True)
        processes.append(process)
        process.start()
        child_conn.close()

    with tqdm.tqdm(total=n_tracks,leave=True) as bar:

        count = 0
        while True:
            for conn in connections:
                obs,track,tracks,info = conn.recv()

                entry = {}
                entry['x'] = any(info['x'])
                entry['t'] = any(info['t'])
                entry['obstacles'] = True
                df = df.append(entry,ignore_index=True)

                index = max(df.index)

                # save obs as png
                frame = obs.astype(np.uint8)
                im = Image.fromarray(frame)
                im.save("tracks/" + str(index) + ".jpeg")

                # save arrays as dic
                dic = {'track':track,'tracks':tracks,'info':info}

                with open('tracks/' + str(index) + '.pkl', 'wb') as handle:
                        pickle.dump(dic, handle, protocol=pickle.HIGHEST_PROTOCOL)

                bar.update(1)
                count += 1
            df.to_csv('tracks/list.csv')
            if count > n_tracks: 
                break

        for process,conn in zip(processes,connections):
            conn.close()
            process.terminate()

if __name__=='__main__':
    n_tracks = int(input("\nHow many tracks to generate? (eg. 10000) "))
    n_cpu    = int(input("How many cpus to run on? (min: 1) "))

    generate_tracks(n_tracks,n_cpu)
