import torch
import torch.nn.functional as F

#############################################
# Differentiable Histogram Counting Method
#############################################
def differentiable_histogram(x, bins=50):
    minbin = torch.min(x)
    maxbin = torch.max(x)
    n_samples = x.shape[0]

    hist_torch = torch.zeros(bins).to(x.device)
    delta = (maxbin - minbin) / bins

    BIN_Table = torch.arange(start=0, end=bins, step=1) * delta
    #print(f"{BIN_Table}")

    h_r = BIN_Table[1].item()
    mask_sub = (x - minbin < h_r).float()
    hist_torch[0] += torch.sum(mask_sub)
    h_r_plus_1 = BIN_Table[bins-1].item()  # h_(r+1)
    mask_plus = (x - minbin >= h_r_plus_1).float()
    hist_torch[bins-1] += torch.sum(mask_plus)
  
    for dim in range(1, bins-1):
      h_r = BIN_Table[dim ].item()   # h_r
      h_r_sub_1 = BIN_Table[dim - 1].item()  # h_(r-1)
      h_r_plus_1 = BIN_Table[dim + 1].item()  # h_(r+1)
      mask = ((x - minbin >= h_r) & (x - minbin < h_r_plus_1)).float()
      hist_torch[dim] += torch.sum(x*mask)

    #    mask = ((x > h_r) & (x <= h_r_plus_1)).float()
        #mask_sub = ((h_r > x) & (x >= h_r_sub_1)).float()
        #mask_plus = ((h_r_plus_1 >= x) & (x >= h_r)).float()
    #    print(f"{dim} h_r_sub_1 {h_r_sub_1}")
    #    print(f"{dim} h_r_plus_1 {h_r_plus_1}")
    #    print(f"{dim} mask {mask}")
        #print(f"{dim} {mask_sub}")
        #print(f"{dim} {mask_plus}")
        #print(f"{dim} {mask}")
        #sub = torch.sum(((x - h_r_sub_1) * mask_sub),dim=-1)
        #plus = torch.sum(((h_r_plus_1 - x) * mask_plus),dim=-1)
        #print(f"{dim} {sub}")
        #print(f"{dim} {plus}")
      
    #    hist_torch[dim] += torch.sum(mask)
        #hist_torch[dim] += torch.sum(((x - h_r_sub_1) * mask_sub), dim=-1)
        #hist_torch[dim] += torch.sum(((h_r_plus_1 - x) * mask_plus), dim=-1)

    #if delta == 0:
    #  failover = torch.ones(bins)/bins
    #  failover.requires_grad = True
    #  return failover
    #hist = (hist_torch/delta)*delta
    #hist = hist_torch/n_samples
    hist = hist_torch/n_samples
    #hist.requires_grad = True
    return hist
    #if torch.sum(hist) == 0:
    #  failover = torch.ones(bins)/bins
    #  failover.requires_grad = True
    #  return failover
    #return hist/torch.sum(hist)

