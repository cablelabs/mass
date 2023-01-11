
import os
import copy
from argparse import ArgumentParser

import torch
#torch.manual_seed(1000)
import torch.nn as nn
from torch import optim
from torch.distributions.multivariate_normal import MultivariateNormal
import torch.nn.functional as F
from differentiable_histogram import differentiable_histogram
from histograms import GaussianHistogram
from histograms import SoftHistogram
from mass_data_utils import get_moments
from validator import validate
import numpy as np


from c_rnn_gan import Generator, Discriminator
import mass_data_utils
import numpy as np
import math
from scipy.special import kl_div

DATA_DIR = 'data'
CKPT_DIR = 'models'

G_FN = 'c_rnn_gan_g.pth'
G_FN_DEFAULT = 'c_rnn_gan_g.pth.DEFAULT'
G_FN_BEST = 'c_rnn_gan_g_best.pth'
D_FN = 'c_rnn_gan_d.pth'

G_LRN_RATE = 0.001
D_LRN_RATE = 0.001
MAX_GRAD_NORM = 5.0
# following values are modified at runtime
MAX_SEQ_LEN = 200
BATCH_SIZE = 32
NUM_FEATS=1
CURRENT_LOW = -1
CURRENT_HIGH_SCORE = -1
LAST_VALIDATION_EPOCH = 0

EPSILON = 1e-40 # value to use to approximate zero (to prevent undefined results)

class GLoss(nn.Module):
    ''' C-RNN-GAN generator loss
    '''
    def __init__(self):
        super(GLoss, self).__init__()

    def forward(self, logits_gen):
        logits_gen = torch.clamp(logits_gen, EPSILON, 1.0)
        batch_loss = -torch.log(logits_gen)

        return torch.mean(batch_loss)

def corr_loss(data_gen, target_corr):
  real_corr = target_corr
  real_batch_sz = data_gen.shape[0]
  gen_cost = 0
  for b in range(0,real_batch_sz):
      batch = data_gen[b,:,:]
      x = batch[:,0]
      y = batch[:,1]
      vx = x - torch.mean(x)
      vy = y - torch.mean(y)
      gen_cost += torch.sum(vx * vy) / (torch.sqrt(torch.sum(vx ** 2)) * torch.sqrt(torch.sum(vy ** 2)))
  gen_cost = gen_cost/real_batch_sz
  print(f"real corr {real_corr} gen corr {gen_cost}")
  return torch.sqrt((gen_cost - real_corr)**2)*real_batch_sz

def corr_cost(data_gen):
  real_batch_sz = data_gen.shape[0]
  gen_cost = 0
  for b in range(0,real_batch_sz):
      batch = data_gen[b,:,:]
      x = batch[:,0]
      y = batch[:,1]
      vx = x - torch.mean(x)
      vy = y - torch.mean(y)
      gen_cost += torch.sum(vx * vy) / (torch.sqrt(torch.sum(vx ** 2)) * torch.sqrt(torch.sum(vy ** 2)))
  return to_cuda(torch.Tensor([gen_cost/real_batch_sz]))


def dist_loss(data_gen, target_dist):
  real_batch_sz = data_gen.shape[0]
  real_batch_len = data_gen.shape[1]
  gen_cost = 0
  maxx = torch.max(data_gen[:,:,0])
  minx = torch.min(data_gen[:,:,0])
  n = real_batch_sz*real_batch_len
  totbins = round(math.sqrt(n))
  bins = torch.zeros(totbins+1)
  for b in range(0,real_batch_sz):
    batch = data_gen[b,:,:]
    for x in range(0,real_batch_len):
       val = batch[x,0]
       tobin = torch.round(totbins*(val - minx)/(maxx-minx))  
       bins[tobin] += 1/n
  for b in range(0, len(target_dist)):
    if b < totbins:
      gen_cost += (target_dist[b] - gen_dist[b])**2
    else:
      gen_cost += torch_round(n/totbins)**2
    
  print(f"gen cost {gen_cost}")
  return gen_cost




class CorrLoss(nn.Module):
    ''' C-RNN-GAN generator loss
    '''
    def __init__(self, alpha=1.0):
        super(CorrLoss, self).__init__()
        self.MSELoss = nn.MSELoss(reduction="mean")
        self.alpha = alpha

    def forward(self, data_real, data_gen, feats_real, feats_gen, target_corr, target_dist):
        mse_loss = self.MSELoss(feats_real, feats_gen)
        closs = corr_loss(data_gen, target_corr)
        loss = self.alpha * closs + (1-self.alpha) * mse_loss
        print(f"Got Gloss {loss}")
        return closs

class DistLoss(nn.Module):
    ''' C-RNN-GAN generator loss
    '''
    def __init__(self, histogram, alpha=1.0):
        super(DistLoss, self).__init__()
        #self.KLDLoss = nn.KLDivLoss(reduction = 'batchmean')
        #self.alpha = alpha
        self.epoch = 0
        self.loss_type = "dist"
        self.period = 50
        self.histogram = histogram
     

    def get_dist(self,data,feature):
        flat_data = data[:,:,feature].flatten()
        gauss = self.histogram(5,min=torch.min(flat_data),max=torch.max(flat_data),sigma=torch.std(flat_data))
        g = gauss(flat_data)
        return g/torch.sum(g)

    def forward(self, data_real, data_gen, feats_real, feats_gen, target_corr, raw_dist):
        if self.epoch > 400:
          loss = corr_loss(data_gen, target_corr)
          self.loss_type = "corr"
        else:
          self.loss_type = "dist"
          #if (int(self.epoch/self.period) % 2) == 0:
          features = data_gen.shape[2]
          loss = 0
          for f in range(0,features):
            gen = self.get_dist(data_gen,f)
            real = self.get_dist(data_real,f)
            loss += torch.sum((real-gen)**2)/50

        self.epoch += 1

        print(f"Got Gloss {self.loss_type} {loss} {self.epoch}")

        real_batch_sz = data_gen.shape[0]
        return loss*real_batch_sz

class MomentsLoss(nn.Module):
    ''' C-RNN-GAN generator loss
    '''
    def __init__(self, moments):
        super(MomentsLoss, self).__init__()
        self.moments = moments
     
    def forward(self, data_real, data_gen, feats_real, feats_gen, target_corr, raw_dist):
        features = data_gen[:,:,:].flatten(start_dim=0,end_dim=1)
        gen = get_moments(features)
        loss = torch.sum((self.moments-gen)**2)

        print(f"Got MomentsGloss {loss}")
        real_batch_sz = data_gen.shape[0]
        return loss*real_batch_sz

def to_cuda(t):
  if torch.cuda.is_available():
    return t.cuda()
  return t

class MultiLoss(nn.Module):
    ''' C-RNN-GAN generator loss
    '''
    def __init__(self, moments, hist):
        super(MultiLoss, self).__init__()
        self.moments = to_cuda(moments)
        self.histogram = GaussianHistogram
        self.epoch = 0
        self.subepoch = 0
        self.max_subepochs = 10
        #self.losses = ["dist","moment","corr"]
        self.losses = ["moment","corr"]
        self.last_loss = -1
        self.hist = hist

    def set_loss(self,loss):
      self.losses = [loss] 
      self.subepoch = 1

    def get_dist(self,data,feature):
        flat_data = to_cuda(data[:,:,feature].flatten())
        gauss = self.histogram(10,min=torch.min(flat_data),max=torch.max(flat_data),sigma=torch.std(flat_data))
        g = gauss(flat_data)
        return g/torch.sum(g)

    def get_dist_loss(self, data_gen, data_real):
       features = data_gen.shape[2]
       loss = 0
       for f in range(0,features):
         gen = self.get_dist(data_gen,f)
         loss += torch.sum((self.hist[f]-gen)**2)
       return loss

    def get_dist_vector(self, data):
       features = data.shape[2]
       d = to_cuda(torch.Tensor())
       for f in range(0,features):
         d = torch.cat((d,self.get_dist(data,f)))
       return d

    def get_gen_vector(self, data_gen):        
        features = data_gen[:,:,:].flatten(start_dim=0,end_dim=1)
        gen = get_moments(features).flatten()
        corr = corr_cost(data_gen)
        dist = self.get_dist_vector(data_gen)
        return torch.cat((gen, corr, dist))

    def get_real_vector(self, data_real, target_corr):        
        dist = self.get_dist_vector(data_real)
        return torch.cat((self.moments, to_cuda(target_corr), dist))
        
    def get_disc_loss(self, logits_gen):
        logits_gen = torch.clamp(logits_gen, EPSILON, 1.0)
        batch_loss = -torch.log(logits_gen)
        return torch.mean(batch_loss) 

    def forward(self, data_real, data_gen, feats_real, feats_gen, target_corr, raw_dist, d_logits_gen):
        if self.subepoch > self.max_subepochs:
          self.subepoch = 0
          #self.losses = ["dist","moment","corr"]
          self.losses = ["moment","corr"]
        if "moment" in self.losses:
          features = data_gen[:,:,:].flatten(start_dim=0,end_dim=1)
          gen = get_moments(features)
          mloss = torch.sum((self.moments-gen)**2)
          self.mloss = mloss.detach()
        else:
          mloss = 0
          self.mloss = 0
        if "corr" in self.losses:
          closs = corr_loss(data_gen, target_corr)
          self.closs = closs.detach()
        else:
          closs = 0
          self.closs = 0 
        if "dist" in self.losses:
          dloss = self.get_dist_loss(data_gen, data_real)
          self.dloss = dloss.detach()
        else:
          dloss = 0
          self.dloss = 0
    
        self.epoch += 1 

        if self.subepoch > 0:
          self.subepoch += 1

        disc_loss = self.get_disc_loss(d_logits_gen)

        print(f"Got MomentsGloss {self.mloss}")
        print(f"Got CorrGloss {self.closs}")
        print(f"Got DiscGloss {disc_loss}")
        loss = mloss + closs + disc_loss
        print(f"Got MultiGloss {loss}")
        real_batch_sz = data_gen.shape[0]
        loss = loss*real_batch_sz 
        self.last_loss = loss.detach()
        return loss


        
class DLoss(nn.Module):
    ''' C-RNN-GAN discriminator loss
    '''
    def __init__(self, label_smoothing=False):
        super(DLoss, self).__init__()
        self.label_smoothing = label_smoothing

    def forward(self, data_real, data_gen, logits_real, logits_gen):
        ''' Discriminator loss

        logits_real: logits from D, when input is real
        logits_gen: logits from D, when input is from Generator

        loss = -(ylog(p) + (1-y)log(1-p))

        '''
        logits_real = torch.clamp(logits_real, EPSILON, 1.0)
        d_loss_real = -torch.log(logits_real)

        if self.label_smoothing:
            p_fake = torch.clamp((1 - logits_real), EPSILON, 1.0)
            d_loss_fake = -torch.log(p_fake)
            d_loss_real = 0.9*d_loss_real + 0.1*d_loss_fake

        logits_gen = torch.clamp((1 - logits_gen), EPSILON, 1.0)
        d_loss_gen = -torch.log(logits_gen)

        batch_loss = d_loss_real + d_loss_gen
        return torch.mean(batch_loss)


def run_training(model, optimizer, criterion, dataloader, freeze_g=False, freeze_d=False):
    ''' Run single training epoch
    '''
    global CURRENT_LOW
    global CURRENT_HIGH_SCORE
    global LAST_VALIDATION_EPOCH
    import numpy as np
   
    target_corr = dataloader.get_corr() 
    target_dist = dataloader.get_dist(MAX_SEQ_LEN) 
    num_feats = dataloader.get_num_mass_features()
    dataloader.rewind(part='train')
    batch_mass = dataloader.get_batch(BATCH_SIZE, MAX_SEQ_LEN, part='train')
    print(f"Batch {batch_mass[0].flatten()}")
    batch_stats = []
    for b in range(0,BATCH_SIZE):
      batch_stats.append(np.correlate(batch_mass[b,:,0],batch_mass[b,:,1])[0])
   
    print(f"Correlate mean {np.mean(batch_stats)} std {np.std(batch_stats)}")
    #cor_bias = np.mean(batch_stats)
    #cor_bias = 0.47
    #eye_init = torch.eye(num_feats) +  abs((torch.eye(num_feats) -1) *-1)* cor_bias
    model['g'].train()
    model['d'].train()

    loss = {}
    g_loss_total = 0.0
    d_loss_total = 0.0
    num_corrects = 0
    num_sample = 0

    while batch_mass is not None:
        real_batch_sz = batch_mass.shape[0]

        # get initial states
        # each batch is independent i.e. not a continuation of previous batch
        # so we reset states for each batch
        # POSSIBLE IMPROVEMENT: next batch is continuation of previous batch
        g_states = model['g'].init_hidden(real_batch_sz)
        d_state = model['d'].init_hidden(real_batch_sz)

        #### GENERATOR ####
        if not freeze_g:
            optimizer['g'].zero_grad()
        # prepare inputs
        z = torch.empty([real_batch_sz, MAX_SEQ_LEN, num_feats]).uniform_() # random vector
        #z = MultivariateNormal(torch.zeros(real_batch_sz,MAX_SEQ_LEN, num_feats), eye_init).sample()
        batch_mass = torch.Tensor(batch_mass)

        # feed inputs to generator
        g_feats, _ = model['g'](z, g_states)

        # calculate loss, backprop, and update weights of G
        if isinstance(criterion['g'], GLoss):
            d_logits_gen, _, _ = model['d'](g_feats, d_state)
            loss['g'] = criterion['g'](d_logits_gen)
        else: # feature matching
            # feed real and generated input to discriminator
            _, d_feats_real, _ = model['d'](batch_mass, d_state)
            d_logits_gen, d_feats_gen, _ = model['d'](g_feats, d_state)
            #loss['g'] = criterion['g'](d_feats_real, d_feats_gen)
            loss['g'] = criterion['g'](batch_mass, g_feats, d_feats_real, d_feats_gen, target_corr, target_dist, d_logits_gen)

        #if loss['g'] < real_batch_sz and criterion['g'].dloss < 0.001 and criterion['g'].mloss < 0.1 and (CURRENT_LOW == -1 or loss['g'] < CURRENT_LOW):
        #if len(criterion['g'].losses) == 3 and loss['g'] < real_batch_sz and criterion['g'].dloss < 0.005 and criterion['g'].mloss < 0.5 and (CURRENT_LOW == -1 or loss['g'] < CURRENT_LOW):
        if (criterion['g'].epoch - LAST_VALIDATION_EPOCH > 1000) or (len(criterion['g'].losses) == 2 and criterion['g'].last_loss < real_batch_sz and (CURRENT_LOW == -1 or criterion['g'].last_loss < CURRENT_LOW)):
          LAST_VALIDATION_EPOCH = criterion['g'].epoch
          print(f"Saving Generator... {criterion['g'].last_loss} epoch {criterion['g'].epoch}")
          if criterion['g'].last_loss < CURRENT_LOW or CURRENT_LOW == -1:
            CURRENT_LOW = criterion['g'].last_loss
          torch.save(model['g'].state_dict(), os.path.join(CKPT_DIR, G_FN))
          try:
            start_time = time.time()
            validation = validate()
            end_time = time.time()
            print(f"Validation time {end_time - start_time}")
          except:
            #validation = [-1,-1,-1]
            validation = [-1,-1]
          val_min = min(validation)
          if CURRENT_HIGH_SCORE == -1 or val_min > CURRENT_HIGH_SCORE:
              CURRENT_HIGH_SCORE = val_min
              print(f"Saving Generator Best Min Validation {val_min} epoch {criterion['g'].epoch}")
              torch.save(model['g'].state_dict(), os.path.join(CKPT_DIR, G_FN_BEST))
          print(f"Saving Generator Validation {validation} epoch {criterion['g'].epoch}")
          maxloss = np.argmin(validation)
          #if maxloss == 0:
          #  criterion['g'].set_loss("dist")
          if maxloss == 0:
            criterion['g'].set_loss("moment")
          elif maxloss == 1:
            criterion['g'].set_loss("corr")



        if not freeze_g:
            loss['g'].backward()
            nn.utils.clip_grad_norm_(model['g'].parameters(), max_norm=MAX_GRAD_NORM)
            optimizer['g'].step()

        #### DISCRIMINATOR ####
        if not freeze_d:
            optimizer['d'].zero_grad()
        # feed real and generated input to discriminator
        d_logits_real, _, _ = model['d'](batch_mass, d_state)
        # need to detach from operation history to prevent backpropagating to generator
        d_logits_gen, _, _ = model['d'](g_feats.detach(), d_state)
        # calculate loss, backprop, and update weights of D
        loss['d'] = criterion['d'](batch_mass, g_feats, d_logits_real, d_logits_gen)
        if not freeze_d:
            loss['d'].backward()
            nn.utils.clip_grad_norm_(model['d'].parameters(), max_norm=MAX_GRAD_NORM)
            optimizer['d'].step()

        g_loss_total += loss['g'].item()
        d_loss_total += loss['d'].item()
        num_corrects += (d_logits_real > 0.5).sum().item() + (d_logits_gen < 0.5).sum().item()
        num_sample += real_batch_sz

        # fetch next batch
        batch_mass = dataloader.get_batch(BATCH_SIZE, MAX_SEQ_LEN, part='train')

    g_loss_avg, d_loss_avg = 0.0, 0.0
    d_acc = 0.0
    if num_sample > 0:
        g_loss_avg = g_loss_total / num_sample
        d_loss_avg = d_loss_total / num_sample
        d_acc = 100 * num_corrects / (2 * num_sample) # 2 because (real + generated)

    return model, g_loss_avg, d_loss_avg, d_acc


def run_validation(model, criterion, dataloader):
    ''' Run single validation epoch
    '''
    num_feats = dataloader.get_num_mass_features()
    dataloader.rewind(part='validation')
    batch_mass = dataloader.get_batch(BATCH_SIZE, MAX_SEQ_LEN, part='validation')

    model['g'].eval()
    model['d'].eval()

    g_loss_total = 0.0
    d_loss_total = 0.0
    num_corrects = 0
    num_sample = 0
    target_corr = dataloader.get_corr()
    while batch_mass is not None:

        real_batch_sz = batch_mass.shape[0]

        # initial states
        g_states = model['g'].init_hidden(real_batch_sz)
        d_state = model['d'].init_hidden(real_batch_sz)

        #### GENERATOR ####
        # prepare inputs
        z = torch.empty([real_batch_sz, MAX_SEQ_LEN, num_feats]).uniform_() # random vector
        batch_mass = torch.Tensor(batch_mass)

        # feed inputs to generator
        g_feats, _ = model['g'](z, g_states)
        # feed real and generated input to discriminator
        d_logits_real, d_feats_real, _ = model['d'](batch_mass, d_state)
        d_logits_gen, d_feats_gen, _ = model['d'](g_feats, d_state)
        # calculate loss
        if isinstance(criterion['g'], GLoss):
            g_loss = criterion['g'](d_logits_gen)
        else: # feature matching
            #g_loss = criterion['g'](d_feats_real, d_feats_gen)
            g_loss = criterion['g'](batch_mass, g_feats, d_feats_real, d_feats_gen, target_corr, d_logits_gen)
        

        d_loss = criterion['d'](batch_mass, g_feats, d_logits_real, d_logits_gen)

        g_loss_total += g_loss.item()
        d_loss_total += d_loss.item()
        num_corrects += (d_logits_real > 0.5).sum().item() + (d_logits_gen < 0.5).sum().item()
        num_sample += real_batch_sz

        # fetch next batch
        batch_mass = dataloader.get_batch(BATCH_SIZE, MAX_SEQ_LEN, part='validation')

    g_loss_avg, d_loss_avg = 0.0, 0.0
    d_acc = 0.0
    if num_sample > 0:
        g_loss_avg = g_loss_total / num_sample
        d_loss_avg = d_loss_total / num_sample
        d_acc = 100 * num_corrects / (2 * num_sample) # 2 because (real + generated)

    return g_loss_avg, d_loss_avg, d_acc


def run_epoch(model, optimizer, criterion, dataloader, ep, num_ep,
              freeze_g=False, freeze_d=False, pretraining=False):
    ''' Run a single epoch
    '''
    model, trn_g_loss, trn_d_loss, trn_acc = \
        run_training(model, optimizer, criterion, dataloader, freeze_g=freeze_g, freeze_d=freeze_d)

    #val_g_loss, val_d_loss, val_acc = run_validation(model, criterion, dataloader)
    val_g_loss = 0
    val_d_loss = 0
    val_acc = 0

    if pretraining:
        print("Pretraining Epoch %d/%d " % (ep+1, num_ep), "[Freeze G: ", freeze_g, ", Freeze D: ", freeze_d, "]")
    else:
        print("Epoch %d/%d " % (ep+1, num_ep), "[Freeze G: ", freeze_g, ", Freeze D: ", freeze_d, "]")

    print("\t[Training] G_loss: %0.8f, D_loss: %0.8f, D_acc: %0.2f\n"
          "\t[Validation] G_loss: %0.8f, D_loss: %0.8f, D_acc: %0.2f" %
          (trn_g_loss, trn_d_loss, trn_acc,
           val_g_loss, val_d_loss, val_acc))
    #g_model = copy.deepcopy(model['g'])
    #g_states = g_model.init_hidden(1)
    #z = torch.empty([1, MAX_SEQ_LEN, NUM_FEATS]).uniform_() # random vector
    #g_feats, g_states = g_model(z, g_states)
    #mass_data = g_feats.squeeze().cpu()
    #mass_data = mass_data.detach().numpy()
    #print(f"\tGenerated {mass_data[0:MAX_SEQ_LEN]}") 




    # -- DEBUG --
    # This is for monitoring the current output from generator
    # generate from model then save to MIDI file

    #g_states = model['g'].init_hidden(1)
    #num_feats = dataloader.get_num_mass_features()
    #z = torch.empty([1, MAX_SEQ_LEN, num_feats]).uniform_() # random vector
    #if torch.cuda.is_available():
    #    z = z.cuda()
    #    model['g'].cuda()

    #model['g'].eval()
    #g_feats, _ = model['g'](z, g_states)


    #song_data = g_feats.squeeze().cpu()
    #song_data = song_data.detach().numpy()

    #if (ep+1) == num_ep:
    #    midi_data = dataloader.save_data('sample.mass', song_data)
    #else:
    #    midi_data = dataloader.save_data(None, song_data)
    #    print(midi_data[0][:16])
    # -- DEBUG --

    return model, trn_acc, trn_g_loss


def main(args):
    ''' Training sequence
    '''
    histogram = SoftHistogram
    if args.gauss_histogram:
      histogram = GaussianHistogram
    dataloader = mass_data_utils.MassDataLoader(args.num_features,args.normalized)
    num_feats = dataloader.get_num_mass_features()

    # First checking if GPU is available
    train_on_gpu = torch.cuda.is_available()
    if train_on_gpu:
        print('Training on GPU.')
    else:
        print('No GPU available, training on CPU.')

    model = {
        'g': Generator(num_feats, use_cuda=train_on_gpu),
        'd': Discriminator(num_feats, use_cuda=train_on_gpu)
    }

    if args.use_sgd:
        optimizer = {
            'g': optim.SGD(model['g'].parameters(), lr=args.g_lrn_rate, momentum=0.9),
            'd': optim.SGD(model['d'].parameters(), lr=args.d_lrn_rate, momentum=0.9)
        }
    else:
        optimizer = {
            'g': optim.Adam(model['g'].parameters(), args.g_lrn_rate),
            'd': optim.Adam(model['d'].parameters(), args.d_lrn_rate)
        }

    criterion = {
        'g': nn.MSELoss(reduction='sum') if args.feature_matching else GLoss(),
        'd': DLoss(args.label_smoothing)
    }
    if args.corr_matching:
        #criterion['g'] = CorrLoss(args.corr_alpha)
        #criterion['g'] = DistLoss(histogram,args.corr_alpha)
        #criterion['g'] = MomentsLoss(dataloader.get_moments())
        criterion['g'] = MultiLoss(dataloader.get_moments(), dataloader.get_hist())

    if args.load_g:
        ckpt = torch.load(os.path.join(CKPT_DIR, G_FN_DEFAULT))
        model['g'].load_state_dict(ckpt)
        print("Continue training of %s" % os.path.join(CKPT_DIR, G_FN_DEFAULT))

    if args.load_d:
        ckpt = torch.load(os.path.join(CKPT_DIR, D_FN))
        model['d'].load_state_dict(ckpt)
        print("Continue training of %s" % os.path.join(CKPT_DIR, D_FN))

    if train_on_gpu:
        model['g'].cuda()
        model['d'].cuda()

    if not args.no_pretraining:
        for ep in range(args.d_pretraining_epochs):
            model, _, _ = run_epoch(model, optimizer, criterion, dataloader,
                              ep, args.d_pretraining_epochs, freeze_g=True, pretraining=True)

        for ep in range(args.g_pretraining_epochs):
            model, _, _ = run_epoch(model, optimizer, criterion, dataloader,
                              ep, args.g_pretraining_epochs, freeze_d=True, pretraining=True)

    freeze_d = False
    freeze_g = False
    for ep in range(args.num_epochs):
        # if ep % args.freeze_d_every == 0:
        #     freeze_d = not freeze_d

        model, trn_acc, gloss = run_epoch(model, optimizer, criterion, dataloader, ep, args.num_epochs, freeze_g=freeze_g, freeze_d=freeze_d)
        if args.conditional_freezing:
            # conditional freezing
            freeze_d = False
            if trn_acc >= 95.0:
                freeze_d = True
            #if gloss == 0:
            #    freeze_g = True

    if not args.no_save_g:
        torch.save(model['g'].state_dict(), os.path.join(CKPT_DIR, G_FN))
        print("Saved generator: %s" % os.path.join(CKPT_DIR, G_FN))
    if os.path.exists(os.path.join(CKPT_DIR, G_FN_BEST)):
      os.rename(os.path.join(CKPT_DIR, G_FN_BEST), os.path.join(CKPT_DIR, G_FN))
    else:
      torch.save(model['g'].state_dict(), os.path.join(CKPT_DIR, G_FN))

    if not args.no_save_d:
        torch.save(model['d'].state_dict(), os.path.join(CKPT_DIR, D_FN))
        print("Saved discriminator: %s" % os.path.join(CKPT_DIR, D_FN))


if __name__ == "__main__":

    ARG_PARSER = ArgumentParser()
    ARG_PARSER.add_argument('--load_g', action='store_true')
    ARG_PARSER.add_argument('--load_d', action='store_true')
    ARG_PARSER.add_argument('--no_save_g', action='store_true')
    ARG_PARSER.add_argument('--no_save_d', action='store_true')

    ARG_PARSER.add_argument('--num_epochs', default=300, type=int)
    ARG_PARSER.add_argument('--num_features', default=1, type=int)
    ARG_PARSER.add_argument('--seq_len', default=256, type=int)
    ARG_PARSER.add_argument('--batch_size', default=16, type=int)
    ARG_PARSER.add_argument('--g_lrn_rate', default=0.001, type=float)
    ARG_PARSER.add_argument('--d_lrn_rate', default=0.001, type=float)

    ARG_PARSER.add_argument('--no_pretraining', action='store_true')
    ARG_PARSER.add_argument('--g_pretraining_epochs', default=5, type=int)
    ARG_PARSER.add_argument('--d_pretraining_epochs', default=5, type=int)
    # ARG_PARSER.add_argument('--freeze_d_every', default=5, type=int)
    ARG_PARSER.add_argument('--use_sgd', action='store_true')
    ARG_PARSER.add_argument('--conditional_freezing', action='store_true')
    ARG_PARSER.add_argument('--label_smoothing', action='store_true')
    ARG_PARSER.add_argument('--feature_matching', action='store_true')
    ARG_PARSER.add_argument('--corr_matching', action='store_true')
    ARG_PARSER.add_argument('--corr_alpha', default=1.0, type=float)
    ARG_PARSER.add_argument('--normalized', action='store_true')
    ARG_PARSER.add_argument('--gauss_histogram', action='store_true')

    ARGS = ARG_PARSER.parse_args()
    MAX_SEQ_LEN = ARGS.seq_len
    BATCH_SIZE = ARGS.batch_size
    NUM_FEATS = ARGS.num_features

    main(ARGS)
