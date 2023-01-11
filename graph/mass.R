#! /usr/bin/env Rscript

source("graph/plotfunctions.R")

png('images/mass.png',4000,10000)
par(mfrow=c(16,10))
for (user in 0:147) {
  plot_mass(user,'data')
}
dev.off()
