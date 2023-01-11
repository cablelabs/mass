#! /usr/bin/env Rscript
library('matrixStats')

args = commandArgs(trailingOnly=TRUE)
num_samples = as.numeric(args[1])
data_dir = args[2]
col1 = as.numeric(args[3])
col2 = as.numeric(args[4])
cors = c()
for (user in 0:(num_samples-1)) {
  fname = paste(data_dir,'/',user,'.mass',sep='')
  data = read.table(fname)
  cors = cbind(cors, cor(data[,col1],data[,col2]))
}
rowMeans(cors)
rowSds(cors)
