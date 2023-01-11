#! /usr/bin/env python3
import sys
infile = sys.argv[1]
idx = sys.argv[2]
import os
import numpy as np
from scipy.stats import percentileofscore

SMOOTH=int(sys.argv[3])

stream_apps = ["MUSIC_AND_AUDIO", 
             "MAPS_AND_NAVIGATION", 
             "SPORTS", 
             "VIDEO_PLAYERS"]


with open(infile) as f:
    raw = [ (float(x.split(" ")[0]),float(x.split(" ")[1]),float(x.split(" ")[2]),float(x.split(" ")[3]),x.split(" ")[4]) for x in f.read().split('\n')[1:-2] ]

rx_avg = []
tx_avg = []
cellsig_avg = []
wifisig_avg = []
apps = []


rx_sd_sum = 0
tx_sd_sum = 0
wifisig_sd_sum = 0

mass = []
mass_app = []
for d in range(0,len(raw)):
  rx = raw[d][0]
  tx = raw[d][1]
  wifisig = raw[d][3]
  app = raw[d][4]

  if len(rx_avg) < SMOOTH:
    rx_avg.append(rx)
    tx_avg.append(tx)
    wifisig_avg.append(wifisig)
    if app != "UNKNOWN":
      apps.append(app)
    continue
  r = sum(rx_avg)/SMOOTH
  t = sum(tx_avg)/SMOOTH
  w = sum(wifisig_avg)/SMOOTH

  rx_sd_sum += np.std(rx_avg)
  tx_sd_sum += np.std(tx_avg)
  wifisig_sd_sum += np.std(wifisig_avg)
 
  if len(apps) > 0:
    a = apps[-1] 
  else:
    a = "UNKNOWN"
  mass.append([r,t,w])
  mass_app.append(a)
  rx_avg = []
  tx_avg = []
  wifisig_avg = []
  apps = []
IGNORE_SIGNAL = os.getenv("IGNORE_SIGNAL")

if rx_sd_sum == 0 or tx_sd_sum == 0 or (IGNORE_SIGNAL != "yes" and wifisig_sd_sum == 0):
  sys.exit(0)

if len(mass) > 50: 
  a = np.array(mass)
  mass_min = a.min(axis=0)
  mass_range = (a.max(axis=0)-a.min(axis=0))

  with open(f"data/{idx}.mass",'w') as f:
    with open(f"data/{idx}.massn",'w') as fn:
      for midx in range(0,len(mass)):
        m = mass[midx]
        m_app = mass_app[midx]
        sep=""
        for i in range(0,3):
          if mass_range[i] == 0:
             val = 0
          else:
             val =  (m[i] - mass_min[i])/mass_range[i] 
          f.write(f"{sep}{m[i]}")
          fn.write(f"{sep}{val}")
          sep=" "
        if m_app in stream_apps:
          category = "STREAM"
        else:
          category = "INTERACT"
        if m[2] < -75:
          signal = "LOW"
        else:
          signal = "HIGH"
        f.write(f" {m_app} {category} {signal}")
        f.write("\n")
        fn.write(f" {m_app} {category} {signal}")
        fn.write("\n")
