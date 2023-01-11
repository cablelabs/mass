#! /usr/bin/env python3
import numpy as np
import os
from mass_data_utils import MassDataLoader

import sys
datadir = sys.argv[1]

m = MassDataLoader(3, False)
idx = 0
fname = f"{datadir}/{idx}.mass"    
x = []
while os.path.isfile(fname):
  data = m.read_one_file(datadir,f"{idx}.mass")
  for d in data:
    if d[2] < -90:
      d[2] = -90
    if d[2] > -30:
      d[2] = -30
    x.append(d)
  idx += 1
  fname = f"{datadir}/{idx}.mass"    

x = np.array(x)
print(f"{x.shape}")

maxx = np.max(x,axis=0)
minx = np.min(x,axis=0)

print(f"max {maxx} min {minx}")

x[:,2] = (x[:,2]-minx[2])/(maxx[2]-minx[2])

avg = [ [] for i in range(2) ]
for v in x:
  if v[2] < .3:
   avg[0].append(v)
  else:
   avg[1].append(v)

a = [ [] for i in range(2) ]
for i in range(2):
  if len(avg[i]) > 0:
    a[i].append(list(np.mean(avg[i],axis=0)))
    print(f"Len {i} = {len(avg[i])}") 
print(f"Avg {a}")
cutoff = (0.3 * (maxx[2]-minx[2])) + minx[2]
print(f"Cutoff {cutoff}")


idx = 0
widx = 0
fname = f"{datadir}/{idx}.mass"    
while os.path.isfile(fname):
   write0 = 0
   write1 = 0
   wname = f"{datadir}/{widx}.mass"    
   with open(fname) as f:
     with open(wname + ".0",'w') as f0:
       with open(wname + ".1",'w') as f1:
             for line in f:
               cols = line.strip('\n').split(" ")
               if float(cols[2]) < cutoff:
                 f0.write(" ".join(cols) + "\n")
                 write0 += 1
               else:
                 f1.write(" ".join(cols) + "\n")
                 write1 += 1
   idx += 1
   fname = f"{datadir}/{idx}.mass"    
   if write0 > 12 and write1 > 12:    
     widx += 1
