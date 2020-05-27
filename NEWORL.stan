// ORL Model
data {
  int<lower=1> T; // Maximum number of trials
  int<lower=1> s;  // Number of subjects
  int<lower=1, upper= T> Tsubj[s]; // Number of trials for each subject
  real o[s,T]; // outcome for each subject on each trial
  real sign[s,T]; // sign for frequency updating for each subject on each trial
  int x[s,T]; // choice for each subject on each trial
}

// Set starting value at zero
transformed data {
  vector[4] initV;
  initV = rep_vector(0.0,4);
}

parameters {
 // Hyper group level parameters
 real<lower = 0, upper = 1> mu_Arew;
 real<lower = 0, upper = 1> mu_Apun;
 real<lower = 0, upper = 5> mu_K;
 real mu_wf;
 real mu_wp;
 
 real<lower=0> sigma_Arew;
 real<lower=0> sigma_Apun;
 real<lower=0> sigma_K;
 real<lower=0> sigma_wf;
 real<lower=0> sigma_wp;
 
 // individual parameters 
 vector<lower=0, upper = 1>[s] Arew;
 vector<lower=0, upper = 1>[s] Apun;
 vector<lower=0, upper = 5>[s] K;
 vector[s] wf;
 vector[s] wp;
}



model {
 // Group level priors
 mu_Arew ~ normal(0.5,1);
 mu_Apun ~ normal(0.5,1);
 mu_K ~ normal(0,1);
 mu_wf ~ normal(0,1);
 mu_wp ~ normal(0,1);
 
 sigma_Arew ~ gamma(1, 1);
 sigma_Apun ~ gamma(1, 1);
 sigma_K ~ gamma(1, 1);
 sigma_wf ~ gamma(1, 1);
 sigma_wp ~ gamma(1, 1);
  
  // individual priors
  Arew ~ normal(mu_Arew, sigma_Arew);
  Apun ~ normal(mu_Apun, sigma_Apun);
  K ~ normal(mu_K, sigma_K);
  wp ~ normal(mu_wp, sigma_wp);
  wf ~ normal(mu_wf, sigma_wf);
  
  
  for (i in 1:s){
    vector[4] EF; // Expected frequency
    vector[4] EV; // Expedcted value
    vector[4] PS; // Perseverance
    vector[4] util; // Combined 'choice value'
    vector[4] PE_EFall; // Prediction error frequency, unchosen
    vector[4] PE_EVall; // Prediction error value, unchosen 
    
    real PE_EF; // Prediction error for expected frequency
    real PE_EV; // Prediciton errror for expected value
    real EF_chosen; // Expected frequency for chosen deck
    real EV_chosen; // Expected value for chosen deck
    real K_tr; // K transformed 
    
    
    // Set starting values to zero
    EF = initV;
    EV = initV;
    PS = initV;
    util = initV;
    K_tr = pow(3, K[i])-1; # Transform K 
    
    
    for (t in 1:Tsubj[i]){
      // Make choice 
      x[i, t] ~ categorical_logit(util); 
      
      // Compute prediction errors
      PE_EV = o[i,t] - EV[x[i,t]];
      PE_EF = sign[i,t] - EF[x[i,t]];
      PE_EFall = -sign[i,t]/3 - EF; # Compute counterfactual prediction error
      
      // Store EF and EV for chosen deck
      EF_chosen = EF[x[i,t]];
      EV_chosen = EV[x[i,t]];
      
      if (o[i,t] >= 0){
        // Update EF for all decks
        EF += Apun[i] * PE_EFall;
        // Update chosen deck
        EF[x[i,t]] = EF_chosen + Arew[i]*PE_EF;
        EV[x[i,t]] = EV_chosen + Arew[i]*PE_EV;
      } else{
        
        EF += Arew[i]*PE_EFall;
        
        EF[x[i,t]] = EF_chosen + Apun[i]*PE_EF;
        EV[x[i,t]] = EV_chosen + Apun[i]*PE_EV;
      }
      
      // Update perseverance
      
      PS[x[i,t]] = 1;
      PS /= (1+ K_tr);
      
      // Calculate utility
      util = EV + EF*wf[i] + PS * wp[i];
    }
  }
}


// Use generated quantities to make predictions based on inferred parameters

generated quantities {
  
  // Log likelihood;
  real log_lik[s];
  
  // Predicted choice
  real y_pred[s,T];
  
  // Set posterior predictions to -1 to avoid NULL values
  for (i in 1:s){
    for (t in 1:T){
     y_pred[i,t] = -1; 
    }
  }
  
  { // Local section (saves time)
    for (i in 1:s){
    // Redefine values for the forward simulation  
    vector[4] EF; // Expected frequency
    vector[4] EV; // Expedcted value
    vector[4] PS; // Perseverance
    vector[4] util; // Combined 'choice value'
    vector[4] PE_EFall; // Prediction error frequency, unchosen
    vector[4] PE_EVall; // Prediction error value, unchosen 
    
    real PE_EF;
    real PE_EV;
    real EF_chosen;
    real EV_chosen;
    real K_tr; // K transformed
      
    // Starting values
    log_lik[i] = 0;
    EF = initV;
    EV = initV;
    PS = initV;
    util = initV;
    K_tr = pow(3, K[i]-1);
    
    for (t in 1:Tsubj[i]){
      
      # Compute loglikelihood and prediction
      log_lik[i] += categorical_logit_lpmf(x[i,t]| util);
      
      y_pred[i,t] = categorical_rng(softmax(util));
      
      // Compute prediction errors
      PE_EV = o[i,t] - EV[x[i,t]];
      PE_EF = sign[i,t] - EF[x[i,t]];
      PE_EFall = -sign[i,t]/3 - EF;
      
      // store EF and EV for chosen deck
      EF_chosen = EF[x[i,t]];
      EV_chosen = EV[x[i,t]];
      
      if (o[i,t] >= 0){
        // Update EF for all decks
        EF += Apun[i] * PE_EFall;
        // Update chosen deck
        EF[x[i,t]] = EF_chosen + Arew[i]*PE_EF;
        EV[x[i,t]] = EV_chosen + Arew[i]*PE_EV;
      } else{
        
        EF += Arew[i]*PE_EFall;
        
        EF[x[i,t]] = EF_chosen + Apun[i]*PE_EF;
        EV[x[i,t]] = EV_chosen + Apun[i]*PE_EV;
      }
      
      // Update perseverance
      
      PS[x[i,t]] = 1;
      PS /= (1+ K_tr);
      
      // Calculate utility
      
      util = EV + EF*wf[i] + PS * wp[i];
      }
    }
  }
}
