#! /bin/env python3

from gaussian_histogram import GaussianHistogram
from mass_data_utils import MassDataLoader
import sys
import torch
import numpy as np
import torch.nn.functional as F
import random



test_dir = sys.argv[1]
users = int(sys.argv[2])

m = MassDataLoader(1,True)

dist = m.get_dist()
batch = m.get_batch(users,12)
raw = torch.Tensor(batch[:,:,0].flatten())
raw_std = torch.std(raw)

gauss = GaussianHistogram(50,min=torch.min(raw),max=torch.max(raw),sigma=raw_std)
hist = gauss(dist[0])
hist = hist/sum(hist)
sigma=dist[1]

histc = torch.histc(dist[0],bins=50)
histc = histc/sum(histc)


#batch = m.get_batch(users,12)
#raw_dist = differentiable_histogram(torch.Tensor(batch[:,:,0].flatten()),bins=5)

n=""
if test_dir == "test" or test_dir == "data":
  n = "n"
all_data = []

if test_dir != "random":
  for i in range(0,users):
    val =  m.read_one_file(test_dir,f"{i}.mass{n}")
    for v in val:
      all_data.append(v[0])
else:
  all_data = np.random.uniform(size=12*users)

#print(f"{all_data[0:10]}")

test_tensor = torch.Tensor(all_data)
test_tensor.requires_grad = True
test_gauss = GaussianHistogram(50,min=torch.min(test_tensor),max=torch.max(test_tensor),sigma=np.std(all_data))
test_hist = test_gauss(test_tensor)
test_hist = test_hist/torch.sum(test_hist)
test_histc = torch.histc(test_tensor.detach(),bins=50)
test_histc = test_histc/sum(test_histc)

#raw_data = torch.Tensor(batch[:,:,0].flatten()).detach().numpy()
#raw_hist = torch.histc(torch.Tensor(raw_data),min=min(raw_data),max=max(raw_data),bins=5)
#raw_hist = raw_hist/sum(raw_hist)

#print(f"Raw data hist: {hist}" )
#print(f"Test {test_dir} hist: {test_hist}" )
#print(f"Raw data histc: {histc}" )
#print(f"Test {test_dir} histc: {test_histc}" )


loss = sum((hist - test_hist)**2)/50
hist_loss = sum((histc - test_histc)**2)/50

print(f"loss {loss}")
print(f"histc loss {hist_loss}")
