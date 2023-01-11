#! /usr/bin/env Rscript
suppressMessages(library('LaplacesDemon'))
suppressMessages(library('e1071')) 
suppressMessages(library('fitdistrplus'))
suppressMessages(library('forecast'))
suppressMessages(library('gamlss'))
suppressMessages(library('gamlss.dist'))
suppressMessages(library('gamlss.add'))
suppressMessages(library('pracma'))
suppressMessages(library('magrittr'))


call_func <- function(what, data){
    acceptable_args <- data[names(data) %in% (formals(what) %>% names)]
    do.call(what, acceptable_args %>% as.list)
}


options(error=traceback) 
options(show.error.locations = TRUE)

to_se <- function(data.std) {
   # z-distribution 95% confidence bands
   1.96*data.std/sqrt(500)
}

draw_se <- function(data.mean, data.std, x, color) {
  se = to_se(data.std)
  alpha = adjustcolor(color,alpha.f=0.2)
  ylow = data.mean - se
  yhigh = data.mean + se
  lines(x,data.mean,col=color)
  polygon(c(x,rev(x)), c(ylow,rev(yhigh)),border=NA, col=alpha)
}
plot_mass <- function(user, path) {
  data = read.table(paste(path,'/',user,'.mass',sep=''))
  data = tail(data,n=24)
  maxy.rx = max(data[,1])
  miny.rx = min(data[,1])
  maxy.tx = max(data[,3])
  miny.tx = min(data[,3])
  n = length(data[,1])
  x=1:n
  par(mar=c(5, 4, 4, 6) + 0.1)
  plot(x,data[,1],log="",type='n',ylim=c(miny.rx,maxy.rx),ylab="", axes=F,xlab="")
  axis(2,ylim=c(miny.rx,maxy.rx),col="black",las=1)
  mtext("RX",side=2,line=2.5)
  draw_se(data[,1],data[,2],x,"black")
  box()
  par(new=TRUE)
  plot(x,data[,3],log="",type='n',ylim=c(miny.tx,maxy.tx),ylab="", axes=F, xlab="")
  axis(4,ylim=c(miny.tx,maxy.tx),col="red",col.axis="red",las=1)
  mtext("TX",side=4,col="red",line=4)
  draw_se(data[,3],data[,4],x,"red")
}

plot_simple <- function(user, path, col="black", last=120) {
  data = read.table(paste(path,'/',user,'.mass',sep=''))
  data = tail(data,n=last)
  maxy.rx = max(data[,1])
  miny.rx = min(data[,1])
  n = length(data[,1])
  x=1:n
  par(mar=c(5, 4, 4, 6) + 0.1)
  plot(x,data[,1],log="",type='n',ylim=c(miny.rx,maxy.rx),ylab="", axes=F,xlab="")
  axis(2,ylim=c(miny.rx,maxy.rx),col=col,las=1)
  mtext("RX",side=2,line=2.5)
  lines(x,data[,1],col=col)
}

get_param <- function(model,param) {
  if (all(names(model$coef) != param)) {
      return(0)
  } 
  return(model$coef[names(model$coef) == param])
}

get_params <- function(model) {
  params = c()
  params = append(params,get_param(model,"intercept"))
  params = append(params,get_param(model,"ar1"))
  params = append(params,get_param(model,"ar2"))
  params = append(params,get_param(model,"ar3"))
  params = append(params,get_param(model,"ma1"))
  params = append(params,get_param(model,"ma2"))
  params = append(params,get_param(model,"ma3"))
  params
}
min_distance <- function(test_acfs,num_test, train_acfs,num_train) {
  distance = c()
  for (train_user in 1:num_train) {
    min_distance = -1
    for (test_user in 1:num_test) {
      dist = sqrt(sum((train_acfs[,train_user] - test_acfs[,test_user]) ^ 2))
      if ((dist < min_distance) | (min_distance == -1)) {
        min_distance = dist
      }
    }
    distance = c(distance,min_distance)
  }
  sum(distance)
}
score_sequence <- function(num_train,train_dir,num_test,test_dir,cols=1,last=12) {
  result = c()
  for (i in 1:cols) {
    result = cbind(result,score_sequence_col(num_train, train_dir, num_test, test_dir, col=i, cols=cols,last=last))
  }
  result
}
score_sequence_col <- function(num_train,train_dir,num_test,test_dir,col=1,cols=1,last=12) {
  train_data = c()
  train_data_norm = c()
  train_acfs = c()
  train_cor = c()
  train_streams = c()
  full = Sys.getenv("FULL")
  doFull = T
  if (full == "no") {
    doFull = F
  }
  ctx = Sys.getenv("CONTEXT")
  if (ctx != "") {
    ctx = paste(".",ctx,sep="")
  }
  trainingrows = Sys.getenv("TROWS")
  if (trainingrows != "") {
    trainingrows = as.numeric(trainingrows)
  } else {
    trainingrows = -1
  }

  cat("Loading train users..","\n")
  for (user in 0:(num_train-1)) {
    if (trainingrows == -1) {
      d = read.table(paste(train_dir,'/',user,'.mass',ctx,sep=''))
    } else {
      d = read.table(paste(train_dir,'/',user,'.mass',ctx,sep=''),nrows=trainingrows)
    }
    train_data = append(train_data,d[,col])
    d_norm = read.table(paste(train_dir,'/',user,'.massn',ctx,sep=''))
    train_data_norm = append(train_data_norm,d_norm[,col])

    train_acfs = cbind(train_acfs,pacf(d[,col])$acf[1:10])
    train_streams = cbind(train_streams,d[,col])
    if (col == 1) {
      cord = cor(d[,1:cols])
      upperd = cord[upper.tri(cord)]
      train_cor = cbind(train_cor,upperd)
    }
  }
  cat("Loaded train users","\n")
  cat("Calculating Hurst...","\n")
  train_hurst = 0
  #train_hurst = hurstexp(train_data,display=F)$Hs
  cat("Calculating Density..","\n")
  train_dens = density(train_data,n=10)
  if (col == 1) {
    train_cross = rowMeans(train_cor)
    cat("Train Cross",train_cross,"\n")
  }

  test_data = c()
  test_data_norm = c()
  test_streams = c()
  test_acfs = c()
  test_cor = c()
  cat("Loading test users...","\n")
  for (user in 0:(num_test-1)) {
    d = read.table(paste(test_dir,'/',user,'.mass',sep=''))
    test_data = append(test_data,d[,col])
    d_norm = read.table(paste(test_dir,'/',user,'.massn',sep=''))
    test_data_norm = append(test_data_norm,d_norm[,col])
    test_acfs = cbind(test_acfs,pacf(d[,col])$acf[1:10])
    test_streams = cbind(test_streams,d[,col])
    if (col == 1) {
      cord = cor(d[,1:cols])
      upperd = cord[upper.tri(cord)]
      test_cor = cbind(test_cor,upperd)
    }
  }
  cat("Loaded test users","\n")
  if (col == 1) {
    test_cross = rowMeans(test_cor)
    cat("Test Cross",test_cross,"\n")
  }

  #train_ccfs = c()
  #for (u1 in 1:num_train) {
  #  train.ccf = 0
  #  for (u2 in 1:num_train) {
  #    if (u1 == u2) {
  #      next
  #    }
  #    train.ccf = train.ccf + max(ccf(train_streams[1:12,u1],train_streams[1:12,u2])$acf) 
  #  }
  #  train_ccfs = append(train_ccfs, train.ccf/(num_train-1))
  #}


  test_novelty = 0
  if (doFull) {  
  test_ccfs = c()
  for (u1 in 1:num_test) {
    test.ccf = 0
    for (u2 in 1:num_test) {
      if (u1 == u2) {
        next
      }
      test.ccf = test.ccf + max(ccf(test_streams[,u1],test_streams[,u2])$acf) 
    }
    test_ccfs = append(test_ccfs, test.ccf/(num_test-1))
  }
  test_novelty = 1-mean(test_ccfs)
  }

  #train_novelty= 1-mean(train_ccfs)
  #cat("Train Novelty",train_novelty,"\n")
  cat("Test Novelty Col",col,test_novelty,"\n")
 
  test_hurst = 0 
  #test_hurst = hurstexp(test_data,display=F)$Hs
  test_dens = density(test_data,n=10)

  acf_dist = min_distance(test_acfs,num_test,train_acfs,num_train)

  if (col == 1 && doFull) {
    png('train.png',4000,10000)
    par(mfrow=c(ceiling(num_train/10),10))
    for (user in 0:(num_train-1)) {
      plot_simple(user,train_dir,last=last)
    }
    dev.off()
    png('test.png',4000,10000)
    par(mfrow=c(ceiling(num_test/10),10))
    for (user in 0:(num_test-1)) {
      plot_simple(user,test_dir,col="red",last=last)
    }
    dev.off()
  }


  m_train =c()
  m_test = c()

  train_mean = mean(train_data_norm)
  train_sd = sd(train_data_norm)
  train_skew = skewness(train_data_norm)
  test_mean = mean(test_data_norm)
  test_sd = sd(test_data_norm)
  test_skew = skewness(test_data_norm)
  m_train = c(train_mean,train_sd,train_skew)
  cat("Train Moments Col",col,train_mean,train_sd,train_skew,"\n")
  m_test = c(test_mean,test_sd,test_skew)
  cat("Test Moments Col",col,test_mean,test_sd,test_skew,"\n")
  #for (m in c(1,3)) { 
  #  train_moment = moment(train_data_norm, order=m, center=F)/train_sd**m
  #  test_moment = moment(test_data_norm, order=m, center=F)/test_sd**m
  #  m_train = append(m_train,train_moment)
  #  m_test = append(m_test,test_moment)
  #  cat("Moment",m,":",train_moment, test_moment,'\n')
  #}
  moment_dist = sqrt(sum((m_train - m_test) ^ 2))
  if (col == 1) {
    cross = sqrt(sum((train_cross - test_cross) ^ 2))
  }
  kld = KLD(train_dens$y, test_dens$y)$mean.sum.KLD
  c(kld,acf_dist,moment_dist)
  summary = numeric(0)
  attr(summary,"kld") = kld
  attr(summary,"hurst") = abs(train_hurst-test_hurst)
  attr(summary,"acf") = acf_dist
  attr(summary,"moment") = moment_dist
  #attr(summary,"novelty") = abs(train_novelty - test_novelty)
  attr(summary,"novelty") = test_novelty
  if (col == 1) {
    attr(summary,"cross") = cross
  } else {
    attr(summary,"cross") = 0
  }
  attributes(summary)
}

fit_weibull <- function(num_train=150,train_dir='data',last=120) {
  train_data = c()
  for (user in 0:(num_train-1)) {
    d = read.table(paste(train_dir,'/',user,'.mass',sep=''))
    train_data = append(train_data,d[,1])
  }
  dist = fitdist(traindata, "weibull")
  shape=as.numeric(gsub("","",dist$estimate)[1])
  scale=as.numeric(gsub("","",dist$estimate)[2])
  summary = numeric(0)
  attr(summary,"shape") = shape
  attr(summary,"scale") = scale
  attributes(summary)
}

randomRows = function(df,n){
   return(df[sample(nrow(df),n),])
}

fit_distribution <- function(num_train=150,train_dir='data',col=1,last=120) {
  train_data = c()
  do_sample = Sys.getenv("SAMPLE")
  for (user in 0:(num_train-1)) {
    d = read.table(paste(train_dir,'/',user,'.mass',sep=''))
    if (do_sample != "") {
      d = randomRows(d,100)
    }
    train_data = append(train_data,d[,col])
  }
  options(show.error.messages = F)
  m = fitDist(train_data,trace=F)
  options(show.error.messages = T)
  m
}

fit_mean_std <- function(num_train=150,train_dir='data',col=1,last=120) {
  train_data = c()
  for (user in 0:(num_train-1)) {
    d = read.table(paste(train_dir,'/',user,'.mass',sep=''))
    train_data = append(train_data,d[,col])
  }
  summary = numeric(0)
  attr(summary,"mu") = mean(train_data)
  attr(summary,"sigma") = sd(train_data)
  attributes(summary)
}



fit_normal <- function(num_train=150,train_dir='data',last=120) {
  train_data = c()
  train_acfs = numeric(11)
  for (user in 0:(num_train-1)) {
    d = read.table(paste(train_dir,'/',user,'.mass',sep=''))
    train_data = append(train_data,d[,1])
  }
  dist = fitdist(train_data, "norm")
  mu=as.numeric(gsub("","",dist$estimate)[1])
  stdev=as.numeric(gsub("","",dist$estimate)[2])
  summary = numeric(0)
  attr(summary,"mu") = mu
  attr(summary,"stdev") = stdev
  attributes(summary)
}

fit_uniform <- function(num_train=150,train_dir='data',col=1,last=120) {
  train_data = c()
  train_acfs = numeric(11)
  for (user in 0:(num_train-1)) {
    d = read.table(paste(train_dir,'/',user,'.mass',sep=''))
    train_data = append(train_data,d[,col])
  }
  dist = fitdist(train_data, "unif")
  umin=as.numeric(gsub("","",dist$estimate)[1])
  umax=as.numeric(gsub("","",dist$estimate)[2])
  summary = numeric(0)
  attr(summary,"umin") = umin
  attr(summary,"umax") = umax
  attributes(summary)
}

fit_arima <- function(num_train=150,train_dir='data',last=120) {
  models = c()
  for (user in 0:(num_train-1)) {
    cat("ARIMA training user",user,"\n")
    d = read.table(paste(train_dir,'/',user,'.mass',sep=''))
    t = ts(d[,1],frequency=24)
    model = auto.arima(t,lambda="auto",approximation=T,stepwise=T,seasonal=F,nmodels=10,d=0,D=0)
    models = cbind(models,model)
  }
  models
}



gen_normal <- function(model, users=150, samples=1000) {
  data = c()
  for (user in 1:users) {
    user_data = rnorm(samples,model$mu,model$stdev)
    data = cbind(data,user_data)
  }
  data
}
gen_uniform <- function(model, users=150, samples=1000) {
  data = c()
  for (user in 1:users) {
    user_data = runif(samples,min=model$umin,max=model$umax)
    data = cbind(data,user_data)
  }
  data
}

gen_weibull <- function(model, users=150, samples=1000) {
  data = c()
  for (user in 1:users) {
    user_data = rweibull(samples,model$shape,model$scale)
    data = cbind(data,user_data)
  }
  data
}
gen_distribution <- function(model, users=150, samples=1000) {
  data = c()
  family = model$family[1]
  params=paste("call_func(r",family,",list(n=samples,mu=model$mu,sigma=model$sigma,nu=model$nu,tau=model$tau))",sep="")
  for (user in 1:users) {
    user_data = eval(parse(text=params))
    data = cbind(data,user_data)
  }
  data
}

gen_arima <- function(model, users=150, samples=1000) {
  data = c()
  for (user in 1:users) {
    user_data = arima.sim(model[,user]$model,n=samples)
    data = cbind(data,user_data)
  }
  data
}


