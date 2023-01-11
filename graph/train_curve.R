#! /usr/bin/Rscript

suppressMessages(library(zoo))

d = read.table('results/train_curve.dat')

x = rollmean(d[,1],10)

cory = rollmean(d[,2],10)
corysd=rollapply(d[,2], 10, sd)
momy = rollmean(d[,3],10)
momysd=rollapply(d[,3], 10, sd)
novy = rollmean(d[,4],10)
novysd=rollapply(d[,4], 10, sd)
dury = rollmean(d[,5],10)
durysd=rollapply(d[,5], 10, sd)

draw_se <- function(x,mu, std, color,n=10) {
  se = std/sqrt(n)
  alpha = adjustcolor(color,alpha.f=0.2)
  ylow = mu - se
  yhigh = mu + se
  polygon(c(x,rev(x)), c(ylow,rev(yhigh)),border=NA, col=alpha)
}

plot_curve <- function(name,x,y,ysd,ylab,col) {
png(paste('results/',name,'curve.png',sep=''))
plot(x,y,type="l",ylab=ylab,xlab="Epochs",col=col,
     cex.main=1.5, cex.lab=1.5, cex.axis=1.5)
draw_se(x,y,ysd,col)
dev.off()
}


plot_curve("cor",x,cory,corysd,"Correlation Distance","red")
plot_curve("mom",x,momy,momysd,"Moments Distance","blue")
plot_curve("nov",x,novy,novysd,"Novelty","green")
plot_curve("dur",x,dury,durysd,"Training Duration (s)","black")

