data {
  int<lower=1> J; // Number of countries 
  int<lower=1> maxN; // max number of observations per country
  array[J] int<lower=1> N ; // number of observations per country
  array[J,maxN] int<lower=0> y; // Poisson observations (dummy-filled ragged array)
  int<lower=0> S;
  array[J,maxN] int<lower=0, upper=S> site;
  array[S] int<lower=1, upper=J> country_of_site;      
}

transformed data {
  int<lower=1> eta_start[J] ;
  int<lower=1> eta_end[J] ;
  eta_start[1] = 1 ;
  eta_end[1] = N[1] ;
  for(j in 2:J){
    eta_start[j] = eta_end[j-1] +1 ;
    eta_end[j] = eta_start[j] -1 + N[j] ;
}
}

parameters {
  real<lower=0.> lambda[J]; // each lambda drawn from gamma(a,b);
  matrix[2,J] country_gp_par_scaled_deviations ;   //non-centered std of length scale
  vector[S] site_scaled_deviations ;   //non-centered std of length scale
    
  real<lower=0> site_m_mu;
  real<lower=0> site_s_mu;

  real<lower=0> site_m_sd;
  real<lower=0> site_s_sd;
}

transformed parameters {
  // Non-centered parameterization of per-subject parameters
  vector[J] country_site_mean = exp(log(site_m_mu) + site_s_mu * country_gp_par_scaled_deviations[1]');
  vector[J] country_site_sd = exp(log(site_m_sd) + site_s_sd * country_gp_par_scaled_deviations[2]'); 

  vector<lower=0>[S] site_lambda;
  for (s in 1:S) {
    site_lambda[s] = exp(log(country_site_mean[country_of_site[s]]) + country_site_sd[country_of_site[s]] * site_scaled_deviations[s]);
}
}                                  
                                  

model {    
  target += gamma_lpdf(site_m_mu | 2, 0.5);
  target += gamma_lpdf(site_s_mu | 2, 0.5);
  target += gamma_lpdf(site_m_sd | 2, 0.5);
  target += gamma_lpdf(site_s_sd | 2, 0.5);


  // Subject-level parameters drawn from pop-level distributions
  // (non-centered parameterizations)
  target += std_normal_lpdf(country_gp_par_scaled_deviations[1]) ;
  target += std_normal_lpdf(country_gp_par_scaled_deviations[2]) ;
  target += std_normal_lpdf(site_scaled_deviations) ;

    
    
    
    
  for (j in 1:J) {
      target += poisson_lpmf(y[j,1:N[j]] | site_lambda[site[j,1:N[j]]]); 
  }
}


generated quantities {      
  int<lower=0> count = 0;
  vector[eta_end[J]] log_lik;
  for (j in 1:J) {
      for (n in 1:N[j]){
          count += 1;
          log_lik[count] = poisson_lpmf(y[j,n] | site_lambda[site[j, n]]);
      }
  }
}