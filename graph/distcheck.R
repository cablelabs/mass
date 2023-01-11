#! /usr/bin/env Rscript
suppressMessages(library('LaplacesDemon'))


args = commandArgs(trailingOnly=TRUE)
num_samples = as.numeric(args[1])
train_dir = args[2]
test_dir = args[3]
col = as.numeric(args[4])
train_data = c()
for (user in 0:(num_samples-1)) {
  fname = paste(train_dir,'/',user,'.mass',sep='')
  data = read.table(fname)
  train_data = append(train_data,data[,col])
}
test_data = c()
for (user in 0:(num_samples-1)) {
  fname = paste(test_dir,'/',user,'.mass',sep='')
  data = read.table(fname)
  test_data = append(test_data,data[,col])
}

train_dens = density(train_data,n=50)

test_dens = density(test_data,n=50)

random_dens = density(runif(num_samples*12),n=50)

test_kld = KLD(train_dens$y, test_dens$y)$mean.sum.KLD
random_kld = KLD(train_dens$y, random_dens$y)$mean.sum.KLD

cat("TEST:",test_kld,"\n")
cat("RANDOM:",random_kld,"\n")
