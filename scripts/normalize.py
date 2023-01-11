#! /usr/bin/env python3

from mass_data_utils import MassDataLoader
import sys
import os
import numpy as np

data_dir = sys.argv[1]
rewrite = False
if len(sys.argv) > 2 and sys.argv[2] == "rewrite":
  rewrite = True
FEATURES=2
loader = MassDataLoader(FEATURES,False)

context_env = os.getenv("CONTEXT")
if context_env is None or context_env == "":
  context = ""
else:
  context = "." + context_env

idx = 0
fname = f"data/{idx}.mass{context}"    
maxx = np.zeros(FEATURES)
minx = np.zeros(FEATURES)
while os.path.isfile(fname):
  x = []
  data = loader.read_one_file(f"{data_dir}",f"{idx}.mass{context}")
  for m in data:
    x.append(m)
  x = np.array(x)
  maxx = np.max(x,axis=0)
  minx = np.min(x,axis=0)
  rangex = maxx - minx
  with open(f"{data_dir}/{idx}.denorm{context}",'w') as d:
    sep = ""
    for maxn in maxx:
      d.write(f"{sep}{maxn}")
      sep = " "
    sep = ""
    d.write("\n")
    for minn in minx:
      d.write(f"{sep}{minn}")
      sep = " "
    d.write("\n")
  xns = (x - minx) / rangex
  np.savetxt(f"{data_dir}/{idx}.massn{context}",xns) 
  idx += 1
  fname = f"{data_dir}/{idx}.mass{context}"    

x = np.array(x)

maxx = maxx/idx
minx = minx/idx


if rewrite:
  rangex = maxx-minx

  normalized = (x-minx)/rangex
  idx = 0
  tot = 0
  fname = f"{data_dir}/{idx}.mass{context}"    
  while os.path.isfile(fname):
    data = loader.read_one_file(f"{data_dir}",f"{idx}.mass{context}")
    with open(f"{data_dir}/{idx}.massn{context}",'w') as out:
      for m in data:
        features = normalized[tot]
        tot += 1 
        sep = ""
        for feature in features:
          out.write(f"{sep}{feature}")
          sep = " "
        out.write('\n')
    idx += 1
    fname = f"{data_dir}/{idx}.mass{context}"    
  
with open(f"{data_dir}/denorm{context}",'w') as d:
  sep = ""
  for maxn in maxx:
    d.write(f"{sep}{maxn}")
    sep = " "
  sep = ""
  d.write("\n")
  for minn in minx:
    d.write(f"{sep}{minn}")
    sep = " "
  d.write("\n")
