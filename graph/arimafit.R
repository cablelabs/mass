#! /usr/bin/env Rscript
source("graph/plotfunctions.R")


args = commandArgs(trailingOnly=TRUE)

num_train = as.numeric(args[1])
train_dir = args[2]
model = fit_arima(num_train=num_train,train_dir=train_dir)

data = gen_arima(model,users=num_train)
for (user in 0:(num_train-1)) {
  fname = paste('arima/',user,'.mass',sep='')
  write(data[,(user+1)],file=fname, ncolumns=1)
}
