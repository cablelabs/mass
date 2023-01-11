#! /usr/bin/env python3
import numpy as np
import os

import sys
datadir = sys.argv[1]

#comm_apps = ["MUSIC_AND_AUDIO", 
#             "ENTERTAINMENT", 
#             "MAPS_AND_NAVIGATION", 
#             "COMMUNICATION", 
#             "VIDEO_PLAYERS",
#             "SPORTS", 
#             "SOCIAL"] 
comm_apps = ["MUSIC_AND_AUDIO", 
             "MAPS_AND_NAVIGATION", 
             "SPORTS", 
             "VIDEO_PLAYERS"]

def is_comm(app):
  #return app in comm_apps or app.startswith("GAME")
  return app in comm_apps

idx = 0
widx = 0
fname = f"{datadir}/{idx}.mass"    
rx0 = []
tx0 = []
rx1 = []
tx1 = []
while os.path.isfile(fname):
   write0 = 0
   write1 = 0
   wname = f"{datadir}/{widx}.mass"    
   with open(fname) as f:
     with open(wname + ".0",'w') as f0:
       with open(wname + ".1",'w') as f1:
             for line in f:
               cols = line.strip('\n').split(" ")
               if is_comm(cols[3]):
                 f0.write(" ".join(cols) + "\n")
                 write0 += 1
                 rx0.append(float(cols[0]))
                 tx0.append(float(cols[1]))
               else:
                 f1.write(" ".join(cols) + "\n")
                 write1 += 1
                 rx1.append(float(cols[0]))
                 tx1.append(float(cols[1]))
   idx += 1
   fname = f"{datadir}/{idx}.mass"    
   if write0 > 12 and write1 > 12:    
     widx += 1

print(f"0 Rx {np.mean(rx0)} {np.std(rx0)}, Tx {np.mean(tx0)} {np.std(tx0)}")
print(f"1 Rx {np.mean(rx1)} {np.std(rx1)}, Tx {np.mean(tx1)} {np.std(tx1)}")
