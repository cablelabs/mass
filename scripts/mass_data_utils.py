import os, math, random, re, sys
import numpy as np
from io import BytesIO
import os.path
import torch
from differentiable_histogram import differentiable_histogram
from histograms import GaussianHistogram
from scipy.stats import moment

MASS_DATA  = 0

debug = ''

file_list = {}

file_list['validation'] = []

file_list['test'] = []

def get_moments(x):
  mean = torch.mean(x,axis=0)
  diffs = x - mean
  var = torch.mean(torch.pow(diffs, 2.0),axis=0)
  std = torch.pow(var, 0.5)
  zscores = diffs / std
  skews = torch.mean(torch.pow(zscores, 3.0),axis=0)
  return torch.cat((mean,std,skews))

def to_cuda(t):
  if torch.cuda.is_available():
    return t.cuda()
  return t

def get_dist(data,feature):
    flat_data = to_cuda(data[:,feature].flatten())
    gauss = GaussianHistogram(10,min=torch.min(flat_data),max=torch.max(flat_data),sigma=torch.std(flat_data))
    g = gauss(flat_data)
    return g/torch.sum(g)

class MassDataLoader(object):

  def __init__(self, num_features, normalized):
    self.pointer = {}
    self.pointer['validation'] = 0
    self.pointer['test'] = 0
    self.pointer['train'] = 0
    context_env = os.getenv("CONTEXT")
    if context_env is None or context_env == "":
      context = ""
    else:
      context = "." + context_env
    self.num_features = num_features
    if normalized:
      self.ext = "massn" + context
    else:
      self.ext = "mass" + context
    self.corr = None
    self.dist = None
    self.hist = None
    self.read_data()
    self.moments = None

  def get_corr(self):
    if not self.corr is None:
       return self.corr
    idx = 0
    fname = f"data/{idx}.{self.ext}"    
    corrs = []
    while os.path.isfile(fname):
      data = self.read_one_file('data',f"{idx}.{self.ext}")
      x = []
      y = []
      for m in data:
        x.append(m[0])
        y.append(m[1])
      x = np.array(x)
      y = np.array(y)
      vx = x - np.mean(x)
      vy = y - np.mean(y)
      corr = np.sum(vx * vy) / (np.sqrt(np.sum(vx ** 2)) * np.sqrt(np.sum(vy ** 2)))
      corrs.append(corr)
      idx += 1
      fname = f"data/{idx}.{self.ext}"    
    corr = np.mean(corrs)
    print(f"Got training corr {corr}") 
    self.corr = corr
    return corr

  def get_dist_hist(self,seqlen):
    if not self.dist is None:
       return self.dist
    idx = 0
    fname = f"data/{idx}.{self.ext}"    
    corrs = []
    x = []
    while os.path.isfile(fname):
      data = self.read_one_file('data',f"{idx}.{self.ext}")
      tot = 0
      for m in data:
        x.append(m[0])
      idx += 1
      fname = f"data/{idx}.{self.ext}"    
    maxx = max(x)
    minx = min(x)
    totbins = round(math.sqrt(idx*seqlen))
    bins = np.zeros(totbins+1)
    n = len(x)

    for val in x:
     tobin = round(totbins*(val - minx)/(maxx-minx))  
     bins[tobin] += 1/n 
    return bins

  def get_hist(self):
    if not self.hist is None:
       return self.hist
    idx = 0
    fname = f"data/{idx}.{self.ext}"    
    x = []
    
    while os.path.isfile(fname):
      data = self.read_one_file('data',f"{idx}.{self.ext}")
      tot = 0
      for m in data:
        x.append(m)
      idx += 1
      fname = f"data/{idx}.{self.ext}"    

    data = torch.Tensor(x)
    self.hist = []
    for i in range(0,self.num_features):
        self.hist.append(get_dist(data,i))
    return self.hist

  def get_dist(self,seqlen=0):
    if not self.dist is None:
       return self.dist
    idx = 0
    fname = f"data/{idx}.{self.ext}"    
    corrs = []
    x = []
    
    while os.path.isfile(fname):
      data = self.read_one_file('data',f"{idx}.{self.ext}")
      tot = 0
      for m in data:
        x.append(m[0])
      idx += 1
      fname = f"data/{idx}.{self.ext}"    
    print(f"Dist Train Length: {len(x)}")
    #self.dist = differentiable_histogram(torch.Tensor(x))
    self.dist = torch.Tensor(x)
    return [self.dist,np.std(x)]

  def get_moments(self):
    if not self.moments is None:
       return self.moments
    idx = 0
    fname = f"data/{idx}.{self.ext}"    
    corrs = []
    x = []
    
    while os.path.isfile(fname):
      data = self.read_one_file('data',f"{idx}.{self.ext}")
      tot = 0
      for m in data:
        x.append(m)
      idx += 1
      fname = f"data/{idx}.{self.ext}"    
    print(f"Dist Train Length: {len(x)}")
    self.moments = get_moments(torch.Tensor(x))
    return self.moments



  def read_data(self):
    self.mass = {}
    self.mass['validation'] = []
    self.mass['test'] = []
    self.mass['train'] = []

    # OVERFIT
    count = 0

    current_path = "data/"
    files = os.listdir(current_path)
    for i,f in enumerate(files):
      if not f.endswith(f".{self.ext}"):
        continue
      # OVERFIT
      if debug == 'overfit' and count > 20: break
      count += 1
      
      if i % 100 == 99 or i+1 == len(files):
        print ( 'Reading files: {}'.format(i+1))
      if os.path.isfile(os.path.join(current_path,f)):
        mass_data = self.read_one_file(current_path, f)
        if mass_data is None:
          continue
        if os.path.join(current_path, f) in file_list['validation']:
          self.mass['validation'].append([mass_data])
        elif os.path.join(current_path, f) in file_list['test']:
          self.mass['test'].append([mass_data])
        else:
          self.mass['train'].append([mass_data])

    random.shuffle(self.mass['train'])
    self.pointer['validation'] = 0
    self.pointer['test'] = 0
    self.pointer['train'] = 0
    # DEBUG: OVERFIT. overfit.
    if debug == 'overfit':
      self.mass['train'] = self.mass['train'][0:1]
      #print (('DEBUG: trying to overfit on the following (repeating for train/validation/test):')
      for i in range(200):
        self.mass['train'].append(self.mass['train'][0])
      self.mass['validation'] = self.mass['train'][0:1]
      self.mass['test'] = self.mass['train'][0:1]
    #print (('lens: train: {}, val: {}, test: {}'.format(len(self.songs['train']), len(self.songs['validation']), len(self.songs['test'])))
    return self.mass

  def read_one_file(self, path, filename):
    mass_data = []
    try:
      if debug:
        print (('Reading {}'.format(os.path.join(path,filename))))
      with open(os.path.join(path,filename)) as f:
        data = f.read().split('\n')[0:-1]
        for line in range(0, len(data)):
           cols = data[line].split(" ")
           features = []
           for val in range(0,self.num_features):
             features.append(float(cols[val]))
           mass_data.append(features)
    except Exception as inst:
      print(inst)
      print ( 'Error reading {}'.format(os.path.join(path,filename)))
      return None

    return mass_data

  def rewind(self, part='train'):
    self.pointer[part] = 0

  def get_batch(self, batchsize, masslength, part='train', normalize=True):
    #print (('get_batch(): pointer: {}, len: {}, batchsize: {}'.format(self.pointer[part], len(self.songs[part]), batchsize))
    if self.pointer[part] > len(self.mass[part])-batchsize:
      batchsize = len(self.mass[part]) - self.pointer[part]
      if batchsize == 0:
      	return None

    if self.mass[part]:
      batch = self.mass[part][self.pointer[part]:self.pointer[part]+batchsize]
      self.pointer[part] += batchsize
      num_mass_features = self.num_features
      batch_mass = np.ndarray(shape=[batchsize, masslength, num_mass_features])

      for s in range(len(batch)):
        massmatrix = np.ndarray(shape=[masslength, num_mass_features])
        begin = 0
        if len(batch[s][MASS_DATA]) > masslength:
          begin = random.randint(0, len(batch[s][MASS_DATA])-masslength)
        matrixrow = 0
        n = begin
        while matrixrow < masslength:
          eventindex = 0
          event = np.zeros(shape=[num_mass_features])
          if n < len(batch[s][MASS_DATA]):
            for f in range(0,num_mass_features):
              event[f] = batch[s][MASS_DATA][n][f]
          massmatrix[matrixrow,:] = event
          matrixrow += 1
          n += 1
        batch_mass[s,:,:] = massmatrix

      return batch_mass

    else:
      raise 'get_batch() called but self.mass is not initialized.'
  
  def get_num_mass_features(self):
    return self.num_features

  def save_data(self, filename, mass_data):
    with open(filename,'w') as f:
     for m in mass_data:
       sep = ""
       for i in range(0, self.num_features):
         f.write(f"{sep}{m[i]}")
         sep = " "
       f.write("\n")
    return mass_data
