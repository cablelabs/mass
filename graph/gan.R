#! /usr/bin/env Rscript

source("graph/plotfunctions.R")

png('images/gan.png',4000,10000)
par(mfrow=c(35,10))
for (user in 1:341) {
  plot_mass(user,'gan')
}
dev.off()
