# This file has been created based on "generate.py"
# by John Byron. The referenced code is available in:
#
#     https://github.com/cjbayron/c-rnn-gan.pytorch



import os
from argparse import ArgumentParser
import numpy as np
import torch
import random

from c_rnn_gan import Generator

CKPT_DIR = 'models'
G_FN = 'c_rnn_gan_g.pth'
MAX_SEQ_LEN = 12
FILENAME = 'sample.mass'
BATCH_SIZE = 64


def denormalize(data):
  batch_sz = data.shape[0]
  for b in range(0,batch_sz):
    batch = data[b,:,:]
    maxs = np.max(batch,axis=0)
    mins = np.min(batch,axis=0)
    ranges = maxs-mins
    with open(f"data/{b}.denorm") as d:
      raw = d.read().split('\n')[0:-1]
      maxs_norm = np.array(list(map(lambda x: float(x), raw[0].split(" "))))
      mins_norm = np.array(list(map(lambda x: float(x), raw[1].split(" "))))
      ranges_norm = maxs_norm - mins_norm
    #new_batch = (((batch - mins)/ranges)*ranges_norm) + mins_norm
    new_batch = (batch * ranges_norm) + mins_norm
    data[b,:,:] = new_batch


def get_corr(data_gen_in):
  data_gen = torch.Tensor(data_gen_in) 
  real_batch_sz = data_gen.shape[0]
  gen_cost = 0
  for b in range(0,real_batch_sz):
      batch = data_gen[b,:,:]
      x = batch[:,0]
      y = batch[:,1]
      vx = x - torch.mean(x)
      vy = y - torch.mean(y)
      gen_cost += torch.sum(vx * vy) / (torch.sqrt(torch.sum(vx ** 2)) * torch.sqrt(torch.sum(vy ** 2)))
  return gen_cost/real_batch_sz

def shuffle_data(d):
  return d
  #r = random.randint(0,d.shape[0])
  #out = np.roll(d,r,axis=0)
  #return out
  #noise = .95+np.random.rand(d.shape[0])/10
  #return np.transpose(noise * np.transpose(out))

def save_mass(filename, data_batch, num_feats, context=""):
  real_batch_sz = data_batch.shape[0]

  corr = get_corr(data_batch)
  print(f"Got corr: {corr}")

  file_prefix = f"gan{context}/"
  file_suffix = ".mass"
 
  #with open(filename,'w') as f:
  for b in range(0,real_batch_sz):
    fname = f"{file_prefix}{b}{file_suffix}"
    with open(fname,'w') as f:
      d = data_batch[b,:,:]
      data = shuffle_data(d)
      for d in data:
        if num_feats == 1:
          f.write(f"{d}\n") 
        else:
          sep = ""
          for i in range(0,num_feats):
            f.write(f"{sep}{d[i]}") 
            sep = " "
          f.write("\n") 
    
def generate(n,num_feats,context="", seq_len=12, batch_size=100, do_save=True):
    ''' Sample MIDI from trained generator model
    '''
    use_gpu = torch.cuda.is_available()
    g_model = Generator(num_feats, use_cuda=use_gpu)
    
    if context == "":
      path = G_FN
    else:
      path = G_FN + "." + context


    if not use_gpu:
        ckpt = torch.load(os.path.join(CKPT_DIR, path), map_location='cpu')
    else:
        ckpt = torch.load(os.path.join(CKPT_DIR, path))

    g_model.load_state_dict(ckpt)

    g_states = g_model.init_hidden(batch_size)
    z = torch.empty([batch_size, seq_len, num_feats]).uniform_() # random vector
    if use_gpu:
        z = z.cuda()
        g_model.cuda()

    #g_model.eval()

    full_song_data = []
    for i in range(n):
        g_feats, g_states = g_model(z, g_states)
        song_data = g_feats.squeeze().cpu()
        song_data = song_data.detach().numpy() 
        full_song_data.append(song_data)

    if len(full_song_data) > 1:
        full_song_data = np.concatenate(full_song_data, axis=0)
    else:
        full_song_data = full_song_data[0]

    print('Full sequence shape: ', full_song_data.shape)
    print('Generated {}'.format(FILENAME))
    #denormalize(full_song_data)
    if do_save:
      save_mass(FILENAME,full_song_data,num_feats,context)
    else:
      return full_song_data


if __name__ == "__main__":
    ARG_PARSER = ArgumentParser()
    # number of times to execute generator model;
    # all generated data are concatenated to form a single longer sequence
    ARG_PARSER.add_argument('-n', default=1, type=int)
    ARG_PARSER.add_argument('--num_features', default=2, type=int)
    ARG_PARSER.add_argument('--batch_size', default=64, type=int)
    ARG_PARSER.add_argument('--seq_len', default=12, type=int)
    ARG_PARSER.add_argument('--context', default="", type=str)
    ARGS = ARG_PARSER.parse_args()
    MAX_SEQ_LEN = ARGS.seq_len
    BATCH_SIZE = ARGS.batch_size

    generate(ARGS.n, ARGS.num_features, ARGS.context, MAX_SEQ_LEN, BATCH_SIZE)
