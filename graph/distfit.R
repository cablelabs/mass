#! /usr/bin/env Rscript
library('LaplacesDemon')
source("graph/plotfunctions.R")


args = commandArgs(trailingOnly=TRUE)

num_train = as.numeric(args[1])
train_dir = args[2]
cols = as.numeric(args[3])
models = c()
for (col in 1:cols) {
  model = fit_distribution(num_train=num_train,train_dir=train_dir,col=col)
  models = cbind(models, model)
}

for (user in 0:(num_train-1)) {
  fname = paste('dist/',user,'.mass',sep='')
  if (file.exists(fname)) {
    next
  }
  user_data = c()
  for (col in 1:cols) {
    data = gen_distribution(models[,col],users=1,samples=12)
    user_data = cbind(user_data,data)
  }
  write(t(user_data),file=fname, ncolumns=cols)
}
