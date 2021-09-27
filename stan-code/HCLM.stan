data {
  int<lower=1> J; // Number of countries 
  int<lower=1> maxN; // max number of observations per country
  array[J] int<lower=1> N ; // number of observations per country
  array[J,maxN] int<lower=0> y; // Poisson observations (dummy-filled ragged array)
  array[J,maxN] real x; // Input  (dummy-filled ragged array)
  int<lower=1> n_tilde;
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
  matrix[4,J] country_gp_par_scaled_deviations ;   //non-centered std of length scale
  matrix[2,S] site_scaled_deviations ;   //non-centered std of length scale
    
  real site_m_mu_a;
  real<lower=0> site_s_mu_a;

  real<lower=0> site_m_sd_a;
  real<lower=0> site_s_sd_a;

  real site_m_mu_b;
  real<lower=0> site_s_mu_b;

  real<lower=0> site_m_sd_b;
  real<lower=0> site_s_sd_b;
}

transformed parameters {
    
  // Non-centered parameterization of per-subject parameters
  vector[J] country_a_mean = site_m_mu_a + site_s_mu_a * country_gp_par_scaled_deviations[1]';
  vector<lower=0>[J] country_a_sd = exp(log(site_m_sd_a) + site_s_sd_a * country_gp_par_scaled_deviations[2]'); 

  vector[S] site_a;
  for (s in 1:S) {
    site_a[s] = country_a_mean[country_of_site[s]] + country_a_sd[country_of_site[s]] * site_scaled_deviations[1,s];
}
    
  // Non-centered parameterization of per-subject parameters
  vector[J] country_b_mean = site_m_mu_b + site_s_mu_b * country_gp_par_scaled_deviations[3]';
  vector<lower=0>[J] country_b_sd = exp(log(site_m_sd_b) + site_s_sd_b * country_gp_par_scaled_deviations[4]'); 

  vector[S] site_b;
  for (s in 1:S) {
    site_b[s] = country_b_mean[country_of_site[s]] + country_b_sd[country_of_site[s]] * site_scaled_deviations[2,s];

  
  }
                               
                                                             
}

model {    
  target += normal_lpdf(site_m_mu_a | 0, 1);
    
  target += gamma_lpdf(site_s_mu_a | 2, 0.5);
    
  target += gamma_lpdf(site_m_sd_a | 2, .5);
    
  target += gamma_lpdf(site_s_sd_a | 2, 0.5);
    

  target += normal_lpdf(site_m_mu_b | 0, 1);
    
  target += gamma_lpdf(site_s_mu_b | 2, 0.5);
    
  target += gamma_lpdf(site_m_sd_b | 2, .5);
    
  target += gamma_lpdf(site_s_sd_b | 2, 0.5);

  // Subject-level parameters drawn from pop-level distributions
  // (non-centered parameterizations)
  target += std_normal_lpdf(country_gp_par_scaled_deviations[1]) ;
  target += std_normal_lpdf(country_gp_par_scaled_deviations[2]) ;
  target += std_normal_lpdf(country_gp_par_scaled_deviations[3]) ;
  target += std_normal_lpdf(country_gp_par_scaled_deviations[4]) ;
  target += std_normal_lpdf(site_scaled_deviations[1]) ;
  target += std_normal_lpdf(site_scaled_deviations[2]) ;

  for (j in 1:J) {
    target += poisson_lpmf(
      y[j,1:N[j]] | exp(site_a[site[j,1:N[j]]] + site_b[site[j,1:N[j]]] .* to_vector(log(x[j,1:N[j]]))));
  }
                                                        
    
}

generated quantities {  
  vector[eta_end[J]] log_lik;
  int<lower=0> count = 0;
  for (j in 1:J) {
    for (n in 1:N[j]) {
      count += 1;
      log_lik[count] =  poisson_lpmf(y[j,n] | exp(site_a[site[j,n]] + site_b[site[j,n]] * log(x[j,n])));
  }      
  }
}