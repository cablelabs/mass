#! /bin/env Rscript
library(MASS)
source("graph/plotfunctions.R")

args = commandArgs(trailingOnly=TRUE)

idx=args[1]
samples=as.numeric(args[2])
bench=args[3]

if (!file.exists(paste(bench,'/beta',sep=''))) {
  model.beta = fit_distribution(num_train=samples,train_dir="data")
  shape = model.beta$shape
  scale = model.beta$scale
  write(c(shape,scale),file=paste(bench,'/beta',sep=''), ncolumns=1)
} else {
  data = read.table(paste(bench,'/beta',sep=''))
  shape = data[1,1]
  scale = data[2,1]
}
cat("Fitting shape",shape,"scale",scale,'\n')

data.raw = read.table(paste(bench,'/',idx,'.prebeta',sep=''))[,1]

n = length(data.raw)
x = 1:n

is.normal = shapiro.test(data.raw)$p.value > 0.05

if (!is.normal) {
  data.raw[data.raw < 0] = 0.00001
  m = boxcox(data.raw~x)
  lambda <- m$x[which.max(m$y)]
  cat("Fit with lambda",lambda,'\n')
  if (lambda != 0) {
    for (i in x) {
      data.raw[i] = ((data.raw[i]^lambda)-1)/lambda
    }
  } else {
    for (i in x) {
      data.raw[i] = ln(data.raw[i])
    }
  }
}

mu = mean(data.raw)
stdev = sd(data.raw)
for (i in x) {
    data.raw[i] = qweibull(pnorm(data.raw[i],mu,stdev),shape=shape,scale=scale)
}

fname = paste(bench,'/',idx,'.mass',sep='')
write(data.raw,file=fname, ncolumns=1)
