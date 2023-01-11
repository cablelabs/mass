#! /usr/bin/env Rscript
suppressMessages(library('zoo'))
suppressMessages(library('forecast'))

args = commandArgs(trailingOnly=TRUE)

idx=args[1]
dir=args[2]
data.raw = read.table(paste(dir,'/',idx,'.raw',sep=''))
data = rollmean(data.raw,k=(3))
data = data[,1]

op <- options(warn=2)
data = ts(data, frequency=(24))
tt = try(HoltWinters(data))
if (is(tt,"try-error")) {
  cat("HoltWinters Error","\n")
} else {
  data = tt$fitted[,1]
  for (d in 1:length(data)) {
    if (data[d] <= 0) {
      data[d] = 0.00001
    }
  }
  write(data,file=paste(dir,'/',idx,'.mass',sep=''), ncolumns=1)
}
options(op)
