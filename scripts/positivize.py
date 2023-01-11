#! /usr/bin/env python3

from mass_data_utils import MassDataLoader
import sys
import os
import numpy as np

data_dir = sys.argv[1]
FEATURES=2
loader = MassDataLoader(FEATURES,False)

context_env = os.getenv("CONTEXT")
if context_env is None or context_env == "":
  context = ""
else:
  context = "." + context_env

idx = 0
fname = f"data/{idx}.mass{context}"    
minx = np.zeros(FEATURES)
while os.path.isfile(fname):
  x = []
  data = loader.read_one_file(f"{data_dir}",f"{idx}.mass{context}")
  for m in data:
    x.append(m)
  x = np.array(x)
  minx = np.min(x,axis=0)
  xns = x - minx
  np.savetxt(f"{data_dir}/{idx}.massn{context}",xns) 
  idx += 1
  fname = f"{data_dir}/{idx}.mass{context}"    
