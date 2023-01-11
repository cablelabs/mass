#! /bin/env Rscript
library(MASS)
source("graph/plotfunctions.R")

args = commandArgs(trailingOnly=TRUE)

idx=args[1]
samples=as.numeric(args[2])
bench=args[3]
cols=as.numeric(args[4])

transform <- function(data.raw, model, norm_trans=F) {
  n = length(data.raw)
  x = 1:n

  if (norm_trans) {
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
  } else {
    fn = ecdf(data.raw)
  }

  data.trans = 0*1:(length(data.raw))
  mu = mean(data.raw)
  stdev = sd(data.raw)
  cat(model[1],model[2],model[3],model[4],model[5],"\n")
  m1 = as.numeric(model[1])
  m2 = as.numeric(model[2])
  m3 = as.numeric(model[3])
  m4 = as.numeric(model[4])
  for (i in x) {
    if (norm_trans) {
      prob = pnorm(data.raw[i],mu,stdev)
    } else {
      prob = fn(data.raw[i])
    }
    if (prob == 0) {
      prob = .1
    }
    if (prob == 1) {
      prob = .9
    }
    cmd = paste("call_func(q",model[5],",list(p=prob,mu=m1,sigma=m2,nu=m3,tau=m4))",sep="")
    data.trans[i] = eval(parse(text=cmd))
  }
  data.trans
}



if (!file.exists(paste(bench,'/dist',sep=''))) {
  params = c()
  for (col in 1:cols) {
    model.dist = fit_distribution(num_train=samples,train_dir="data", col)
    mu = model.dist$mu
    sigma = model.dist$sigma
    nu = model.dist$nu
    tau = model.dist$tau
    family = model.dist$family[1]
    params = cbind(params, c(mu,sigma,nu,tau,family)) 
  }
  write(t(params),file=paste(bench,'/dist',sep=''), ncolumns=cols)
} 

model = read.table(paste(bench,'/dist',sep=''))
data.pre = read.table(paste(bench,'/',idx,'.predist',sep=''))
got_error = F
trans = c()
for (c in 1:cols) {
  cat("Fitting family",model[5,c],"mu",model[1,c],"sigma",model[2,c],"nu", model[3,c], "tau", model[4,c],'\n')
  
  tt = try(transform(data.pre[,c],model[,c]))
  if (is(tt,"try-error")) {
      cat("Transform Error","\n")
      got_error = T
      break
  }
  trans = cbind(trans,tt)
}

if (!got_error) {
  fname = paste(bench,'/',idx,'.mass',sep='')
  write(t(trans),file=fname, ncolumns=cols)
}
