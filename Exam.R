# Set working directory
setwd("C:/Users/Simon/Google Drev/Uni/Data Science")

# Load libraries
pacman::p_load(rstan, bayesplot, loo, bayestestR, ggplot2, cowplot, ggpubr)

# Load in data etc. 
choicedata = read.delim('IGTdata/choice_100.txt', sep = " ")
lossdata = read.delim('IGTdata/lo_100.txt', sep = " ")
win_data = read.delim('IGTdata/wi_100.txt', sep = " ")

# Calculate outcomes
outcome = win_data + lossdata
o = as.matrix(outcome)
sign = sign(o)
x = as.matrix(choicedata) 

# Sample 100 participants from the data frame 
outcome_sub = outcome[sample(nrow(outcome), 100), ]
o_sub = as.matrix(outcome_sub)/100 #scale outcome by 100
sign_sub = sign(o_sub)
rownam = rownames(o_sub)
choice_sub = choicedata[rownam, ]
x = as.matrix(choice_sub)

# Number of trials and subjects
T = 100
s = 100
Tsubj = as.vector(rep(100,100))

# Gather data
data = list(o=o_sub, sign=sign_sub, x=x , s=s, T=T, Tsubj = Tsubj)

# ----------------- NUTS Sampling --------------

# Run NUTS
start_time=Sys.time()
samplefit = stan(file = 'NEWORL.stan', data = data, cores = 4, warmup = 1500, iter = 4000)
end_time = Sys.time()
sample_time = end_time - start_time
print(sample_time) # Get computation time

# plots
plot(samplefit, pars = c("mu_Arew", "mu_Apun", "mu_K", "mu_wp", "mu_wf"))
stan_dens(samplefit, pars = c("mu_Arew", "mu_Apun", "mu_K", "mu_wp", "mu_wf")) + ggtitle("NUTS sampling") + theme(plot.title = element_text(hjust = 0.5))
stan_trace(samplefit, pars = c("mu_Arew", "mu_Apun", "mu_K", "mu_wp", "mu_wf"))

# Posterior predictive accuracy
ext_samp2 <- rstan::extract(samplefit, pars = c("y_pred"))
mean(apply(ext_samp2$y_pred, c(2,3), median) == x) 

# Loglik
Sample_loglik=extract_log_lik(samplefit, parameter_name = "log_lik", merge_chains = TRUE)

# --------------------- Variational inference (meanrank) -----------------------

# Run variational inference
ORL = stan_model(file = 'NEWORL.stan')
start_time=Sys.time()
vi_meanrank=vb(ORL, data= data, tol_rel_obj = 0.001, output_samples = 10000) 
end_time = Sys.time()
vi_time = end_time - start_time
print(vi_time) # Get comptutation time

# To get predictive accuracy for the variational inference model
ext_meanrank <- rstan::extract(vi_meanrank, pars = "y_pred")
mean(apply(ext_meanrank$y_pred, c(2,3), median) == x) 

# Plots of VI fit 
stan_dens(vi_meanrank, pars = c("mu_Arew", "mu_Apun", "mu_K", "mu_wp", "mu_wf")) + ggtitle("ADVI (mean rank)") + theme(plot.title = element_text(hjust = 0.5))
stan_trace(vi_meanrank, pars = c("mu_Arew", "mu_Apun", "mu_K", "mu_wp", "mu_wf")) 
plot(vi_meanrank, pars = c("mu_Arew", "mu_Apun", "mu_K", "mu_wp", "mu_wf"))

# Meanrank loglik
Meanrank_loglik=extract_log_lik(vi_meanrank, parameter_name = "log_lik", merge_chains = TRUE)

#-------------- Variational Inference (fullrank) ---------------

# Run fullrank algorithm
start_time=Sys.time()
vi_fullrank=vb(ORL, data= data, algorithm = "fullrank", tol_rel_obj = 0.001, output_samples = 10000)
end_time = Sys.time()
vi_FR_time = end_time - start_time
print(vi_FR_time)

# Posterior predictive check
ext_fullrank <- rstan::extract(vi_fullrank, pars = "y_pred")
mean(apply(ext_fullrank$y_pred, c(2,3), median) == x) 

# Plots
stan_dens(vi_fullrank, pars = c("mu_Arew", "mu_Apun", "mu_K", "mu_wp", "mu_wf")) + ggtitle("ADVI (full rank)") + theme(plot.title = element_text(hjust = 0.5))
stan_trace(vi_fullrank, pars = c("mu_Arew", "mu_Apun", "mu_K", "mu_wp", "mu_wf")) + ggtitle("ADVI (full rank)") + theme(plot.title = element_text(hjust = 0.5))

# VI fullrank summary
vi_fullrank

# Extract loglikelihood
fullrank_loglik=extract_log_lik(vi_fullrank, parameter_name = "log_lik", merge_chains = TRUE)


#--------------Model comparison----------------

# Using leave one out cross validation
loo_compare(loo(fullrank_loglik), loo(Meanrank_loglik), loo(Sample_loglik))

# ---------- Calculate MAPs and HDI----------
ext_samp = rstan::extract(samplefit)
ext_meanfield = rstan::extract(vi_meanrank)

# Get MAP and 95 % HDI from sampling for individual parameters
samp_maps = data.frame(Arew= c(0), Apun = c(0), K = c(0), wf= c(0), wp = c(0))
samp_hdis = data.frame(Arew= c(0), Apun = c(0), K = c(0), wf= c(0), wp = c(0))
for (n in 11:15){
  for (i in 1:100){
    temp = ext_samp[n]
    map_temp= map_estimate(temp[[1]][,i], precision = 12000)
    samp_maps[i,(n-10)] = map_temp 
    hdi_temp = hdi(temp[[1]][,i], ci = .95)
    hdi_dif = hdi_temp$CI_high - hdi_temp$CI_low
    samp_hdis[i, (n-10)] = hdi_dif   
  }
}

# Get MAP and 95 % HDI from VI for individual parameters
vi_maps = data.frame(Arew= c(0), Apun = c(0), K = c(0), wf= c(0), wp = c(0))
vi_hdis = data.frame(Arew= c(0), Apun = c(0), K = c(0), wf= c(0), wp = c(0))
for (n in 11:15){
  for (i in 1:100){
    temp = ext_meanfield[n]
    map_temp= map_estimate(temp[[1]][,i], precision = 1000)
    vi_maps[i,(n-10)] = map_temp 
    hdi_temp = hdi(temp[[1]][,i], ci = .95)
    hdi_dif = hdi_temp$CI_high - hdi_temp$CI_low
    vi_hdis[i, (n-10)] = hdi_dif   
  }
}

# Make density plots to sum up difference for individual in MAPS and HDI
p1=ggplot()+geom_density(aes(samp_maps$Arew - vi_maps$Arew, fill = "red")) + theme_minimal()+guides(fill=FALSE) + xlab("Arew") + scale_fill_manual(values = "firebrick")
p2=ggplot()+geom_density(aes(samp_maps$Apun - vi_maps$Apun, fill = "red")) + theme_minimal()+guides(fill=FALSE) + xlab("Apun") + scale_fill_manual(values = "firebrick")
p3= ggplot()+geom_density(aes(samp_maps$K - vi_maps$K, fill = "red")) + theme_minimal()+guides(fill=FALSE) + xlab("K") + scale_fill_manual(values = "firebrick")
p4=ggplot()+geom_density(aes(samp_maps$wf - vi_maps$wf, fill = "red")) + theme_minimal()+guides(fill=FALSE) + xlab("wf") + scale_fill_manual(values = "firebrick")
p5=ggplot()+geom_density(aes(samp_maps$wp - vi_maps$wp, fill = "red")) + theme_minimal()+guides(fill=FALSE) + xlab("wp") + scale_fill_manual(values = "firebrick")

p=plot_grid(p1, p2, p3, p4, p5)
p_title = ggdraw() + draw_label("Differences in MAP estimates (NUTS - Meanfield)", fontface='bold')
p_map =plot_grid(p_title, p, ncol=1, rel_heights=c(0.1, 1)) # rel_heights values control title margins

p6 = ggplot() +geom_density(aes(samp_hdis$Arew - vi_hdis$Arew, fill = "bla")) + theme_minimal()+ guides(fill=FALSE) + xlab("Arew") + scale_fill_manual(values = "steelblue")
p7 = ggplot() +geom_density(aes(samp_hdis$Apun - vi_hdis$Apun, fill = "bla")) + theme_minimal()+ guides(fill=FALSE) + xlab("Apun") + scale_fill_manual(values = "steelblue")
p8 = ggplot() +geom_density(aes(samp_hdis$K - vi_hdis$K, fill = "bla")) + theme_minimal()+ guides(fill=FALSE) + xlab("K") + scale_fill_manual(values = "steelblue")
p9 = ggplot() +geom_density(aes(samp_hdis$wf - vi_hdis$wf, fill = "bla")) + theme_minimal()+ guides(fill=FALSE) + xlab("wf") + scale_fill_manual(values = "steelblue")
p10 = ggplot() +geom_density(aes(samp_hdis$wp - vi_hdis$wp, fill = "bla")) + theme_minimal()+ guides(fill=FALSE) + xlab("wp") + scale_fill_manual(values = "steelblue")

p=plot_grid(p6, p7, p8, p9, p10)
p_title = ggdraw() + draw_label("Differences in 95 % HDI estimates (NUTS - Meanfield)", fontface='bold')
p_hdi = plot_grid(p_title, p, ncol=1, rel_heights=c(0.1, 1)) # rel_heights values control title margins

plot_grid(p_map, p_hdi, labels = c("A", "B"), ncol = 1)

# Get mean difference for HDI and MAPS
mean(samp_maps$Arew- vi_maps$Arew)
mean(samp_maps$Apun- vi_maps$Apun)
mean(samp_maps$K- vi_maps$K)
mean(samp_maps$wf- vi_maps$wf)
mean(samp_maps$wp- vi_maps$wp)

mean(samp_hdis$Arew- vi_hdis$Arew)
mean(samp_hdis$Apun- vi_hdis$Apun)
mean(samp_hdis$K- vi_hdis$K)
mean(samp_hdis$wf- vi_hdis$wf)
mean(samp_hdis$wp- vi_hdis$wp)

# -------------- Make density plots for group means ----------

# Extract samples for group level parameters
ext_samp = extract(samplefit, pars = c("mu_Arew", "mu_Apun", "mu_K", "mu_wf", "mu_wp"))
ext_mean = extract(vi_meanrank, pars = c("mu_Arew", "mu_Apun", "mu_K", "mu_wf", "mu_wp"))
ext_full = extract(vi_fullrank, pars = c("mu_Arew", "mu_Apun", "mu_K", "mu_wf", "mu_wp"))

# Make plots
p11=ggplot()+geom_density(aes(ext_samp$mu_Arew, colour = "NUTS")) + geom_density(aes(ext_mean$mu_Arew, colour = "Meanfield")) + geom_density(aes(ext_full$mu_Arew, colour = "Fullrank")) + theme_minimal() + xlab("mu Arew") + theme(legend.position = "none")
p12=ggplot()+geom_density(aes(ext_samp$mu_Apun, colour = "NUTS")) + geom_density(aes(ext_mean$mu_Apun, colour = "Meanfield")) + geom_density(aes(ext_full$mu_Apun, colour = "Fullrank")) + theme_minimal() + xlab("mu Apun") +  guides(colour=FALSE)
p13=ggplot()+geom_density(aes(ext_samp$mu_K, colour = "NUTS")) + geom_density(aes(ext_mean$mu_K, colour = "Meanfield")) + geom_density(aes(ext_full$mu_K, colour = "Fullrank")) + theme_minimal() + xlab("mu K") + guides(colour=FALSE)
p14=ggplot()+geom_density(aes(ext_samp$mu_wf, colour = "NUTS")) + geom_density(aes(ext_mean$mu_wf, colour = "Meanfield")) + geom_density(aes(ext_full$mu_wf, colour = "Fullrank")) + theme_minimal() + xlab("mu wf") + guides(colour=FALSE)
p15=ggplot()+geom_density(aes(ext_samp$mu_wp, colour = "NUTS")) + geom_density(aes(ext_mean$mu_wp, colour = "Meanfield")) + geom_density(aes(ext_full$mu_wp, colour = "Fullrank")) + theme_minimal() + xlab("mu wp") + guides(colour=FALSE)

# Arrange into one plot
ggarrange(p11, p12, p13, p14, p15, common.legend = T, legend = "bottom")
