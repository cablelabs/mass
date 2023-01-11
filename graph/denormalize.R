#! /bin/env Rscript
library(MASS)
source("graph/plotfunctions.R")

args = commandArgs(trailingOnly=TRUE)

idx=args[1]
samples=as.numeric(args[2])
bench=args[3]
cols=as.numeric(args[4])

if (!file.exists(paste(bench,'/denorm',sep=''))) {
  params = c()
  for (col in 1:cols) {
    model.dist = fit_mean_std(num_train=samples,train_dir="data", col)
    mu = model.dist$mu
    sigma = model.dist$sigma
    params = cbind(params, c(mu,sigma)) 
  }
  write(t(params),file=paste(bench,'/denorm',sep=''), ncolumns=cols)
} 

model = read.table(paste(bench,'/denorm',sep=''))
data.pre = read.table(paste(bench,'/',idx,'.predist',sep=''))
got_error = F
trans = c()
for (c in 1:cols) {
  mu = model[1,c]
  s = model[2,c] 
  cat("Fitting mu",mu,"sigma",s,'\n')
  old_mu = mean(data.pre[,c])
  old_std = sd(data.pre[,c]) 
  old_val = (data.pre[,c] - old_mu)/old_std
  new_val = (old_val*s) + mu 
  trans = cbind(trans,new_val)
}

if (!got_error) {
  fname = paste(bench,'/',idx,'.mass',sep='')
  write(t(trans),file=fname, ncolumns=cols)
}
