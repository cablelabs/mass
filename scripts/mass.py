#! /usr/bin/env python3
from generate_mass import generate
import numpy as np
import random

class Mass(object):
  def __init__(self, users, seq_len, do_shuffle=False, normalize="minmax"):
    self.users = users
    self.seq_len = seq_len
    self.do_shuffle = do_shuffle
    self.normalize = normalize
    with open('data/contexts') as f:
      self.contexts = f.read().strip('\n').split(" ")

  def pos_normalize(self,trace):
    return trace - np.min(trace,axis=0)

  def minmax_normalize(self, trace):
    mint = np.min(trace,axis=0)
    ranget = np.max(trace,axis=0)-mint
    return (trace - mint)/ranget

  def shuffle(self, d):
    r = random.randint(0,d.shape[0])
    out = np.roll(d,r,axis=0)
    return out

  def generate(self,context="DEFAULT"):
    if context not in self.contexts:
      context = "DEFAULT"
    trace = generate(1,2,context,self.seq_len,self.users,False)
    if self.normalize == "minmax":
      trace = self.minmax_normalize(trace)
    elif self.normalize == "pos":
      trace = self.pos_normalize(trace)
    if self.do_shuffle:
      trace = self.shuffle(trace)
    return trace
