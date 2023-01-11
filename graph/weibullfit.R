#! /usr/bin/env Rscript
library('LaplacesDemon')
source("graph/plotfunctions.R")


args = commandArgs(trailingOnly=TRUE)

num_train = as.numeric(args[1])
train_dir = args[2]
model = fit_weibull(num_train=num_train,train_dir=train_dir)
data = gen_weibull(model,users=num_train)
for (user in 0:(num_train-1)) {
  fname = paste('weibull/',user,'.mass',sep='')
  if (file.exists(fname)) {
    next
  }
  write(data[,(user+1)],file=fname, ncolumns=1)
}
