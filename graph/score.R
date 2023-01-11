#! /usr/bin/env Rscript
library('LaplacesDemon')
source("graph/plotfunctions.R")
options(digits=10)


args = commandArgs(trailingOnly=TRUE)

num_train = as.numeric(args[1])
train_dir = args[2]
num_test = as.numeric(args[3])
test_dir = args[4]
cols = as.numeric(args[5])

score = score_sequence(num_train,train_dir,num_test,test_dir,cols)
score
klds =c()
acfs = c()
moments = c()
novelties = c()
hursts = c()
for (s in 1:cols) {
  klds = append(klds,score[,s]$kld)
  acfs = append(acfs,score[,s]$acf)
  moments = append(moments,score[,s]$moment)
  novelties = append(novelties,score[,s]$novelty)
  hursts = append(hursts,score[,s]$hurst)
}
cat('RESULT',mean(klds),mean(acfs),mean(moments),mean(novelties),mean(hursts),score[,1]$cross,"\n")
