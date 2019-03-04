data {
	int<lower=1> T; 
	int<lower=1> D; 
	int<lower=1> L;

	int<lower=1> N_obs;
	int<lower=1> N_mis;

	vector[N_obs] x_obs;  // N_obs = T x Kt observations as long list (where Kt is #observed at time t)

	int<lower=1> s[T];  // number of observations at each time point 
	int<lower=0> m[T];  // number of missing at each time point 

	int ii_obs[N_obs];  // indices of observed data as long list 
	int ii_mis[N_mis];  // indices of missing data as long list 

	int pos_obs[T];  // starting index at each time point for x_obs 
	int pos_mis[T];  // starting index at each time point for x_mis
}

transformed data {
	
}

parameters {
	vector[L] mu0;  
	cov_matrix[L] Q0; 

	matrix[L,L] A;  
	cov_matrix[L] Q;  

	matrix[D,L] C; 
	cov_matrix[D] R;

	vector[L] y[T];

	vector[N_mis] x_mis;  // N_mis = T x Kt observations as long list (where Kt is #missing at time t)
}

transformed parameters {
	vector[D] x[T];  

	for (t in 1:T){

		x[t, segment(ii_obs, pos_obs[t], s[t]) ] = segment(x_obs, pos_obs[t], s[t]);

		if (m[t] > 0) {
			x[t, segment(ii_mis, pos_mis[t], m[t]) ] = segment(x_mis, pos_mis[t], m[t]);
		}
	}
}

model {
	y[1] ~ multi_normal(mu0, Q0);

	for (i in 2:T)
		y[i] ~ multi_normal(A * y[i-1], Q); 

	for (t in 1:T) {
		x[t] ~ multi_normal(C * y[t], R);
	}
}
