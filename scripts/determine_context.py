#! /usr/bin/env python3

import sys
import os
datadir = sys.argv[1]

idx = 0
fname = f"{datadir}/{idx}.mass"    
contexts = {}
ups = []
downs = []
while os.path.isfile(fname):
   with open(fname) as f:
     local_contexts = {}
     for line in f:
       cols = line.strip('\n').split(" ")
       context = [cols[4],cols[5],cols[4]+"_"+cols[5]]
       downs.append(float(cols[0]))
       ups.append(float(cols[1]))
       for c in context:
         if c not in contexts:
           contexts[c] = {"users":0, "up": 0, "down": 0}
         if c not in local_contexts:
           local_contexts[c] = {"values":0,"up":0,"down":0}
         local_contexts[c]["values"] += 1
         local_contexts[c]["down"] += float(cols[0])
         local_contexts[c]["up"] += float(cols[1])

     for c in local_contexts.keys():
       if local_contexts[c]["values"] > 12:
         contexts[c]["users"] += 1  
         contexts[c]["up"] += local_contexts[c]["up"]/local_contexts[c]["values"]  
         contexts[c]["down"] += local_contexts[c]["down"]/local_contexts[c]["values"]    
   idx += 1
   fname = f"{datadir}/{idx}.mass"    
ups_mean = sum(ups)/len(ups)
downs_mean = sum(downs)/len(downs)
use_ctx = []
for c in contexts.keys():
  down_avg = contexts[c]['down']/contexts[c]['users']
  up_avg = contexts[c]['up']/contexts[c]['users']
  down_diff = abs(down_avg-downs_mean)/downs_mean
  up_diff = abs(up_avg-ups_mean)/ups_mean
  print(f"{c} {contexts[c]['users']}  avg {down_avg} {up_avg} diff {down_diff} {up_diff}",file=sys.stderr)
  if (up_diff > 0.1 or down_diff > 0.1) and contexts[c]['users'] > 5:
    use_ctx.append(c)

print(" ".join(use_ctx))
