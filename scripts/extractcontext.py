#! /usr/bin/env python3
import numpy as np
import os

import sys
datadir = sys.argv[1]
context = sys.argv[2]
contexts = context.split("_")


idx = 0
widx = 0
fname = f"{datadir}/{idx}.mass"    
while os.path.isfile(fname):
   writec = 0
   wname = f"{datadir}/{widx}.mass"    
   with open(fname) as f:
     with open(wname + f".{context}",'w') as fc:
       for line in f:
         cols = line.strip('\n').split(" ")
         is_match = False
         if (len(contexts) == 1 and (cols[4] == contexts[0] or cols[5] == contexts[0])) or \
            (len(contexts) > 1 and (cols[4] + "_" + cols[5]) == context):
           is_match = True
         if is_match:
           fc.write(" ".join(cols) + "\n")
           writec += 1
   idx += 1
   fname = f"{datadir}/{idx}.mass"    
   if writec > 12:    
     widx += 1
   else:
     os.remove(wname + f".{context}")
