import os
import pickle

import numpy as np
import pandas as pd

from gym.envs.box2d.car_racing import MIN_SEGMENT_LENGHT

def clean_tracks(folder='/tracks/'):
    track_list = pd.read_csv(folder + '/list.csv',index_col=0)

    idx_to_remove = set()
    for idx in track_list.index:
        print(idx)
        dictionary = pickle.load(open(folder + '/' + str(idx) + '.pkl','rb'))
        track  = dictionary['track']
        tracks = dictionary['tracks']
        info   = dictionary['info']

        # check if has intersection at 0
        if info[0]['intersection_id'] != -1:
            idx_to_remove.add(idx)
            continue

        # check if all intersections are longer than MIN_SEGMENT_LENGHT +2
        intersections = np.where((info['intersection_id'] != -1) & (info['track'] > 0))[0]
        intersections -= len(tracks[0])
        diff_intersections = np.array([(intersections[i]-intersections[i-1])%(len(tracks[1])) \
                for i in range(len(intersections))])
        if any((diff_intersections > 1) & (diff_intersections < MIN_SEGMENT_LENGHT +2)):
            idx_to_remove.add(idx)

    print("tracks to delete are:", idx_to_remove)

    track_list.drop(list(idx_to_remove))
    track_list.to_csv(folder + '/list.csv')

    for idx in idx_to_remove:
        os.rename(folder+'/'+str(idx)+'.jpeg',folder+'/____'+str(idx)+'.jpeg')
        os.remove(folder+'/'+str(idx)+'.pkl')
    

if __name__=='__main__':
    clean_tracks(folder='./tracks')
