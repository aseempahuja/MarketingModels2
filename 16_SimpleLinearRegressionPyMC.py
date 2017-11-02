"""estimating mean and std dev of gaussian likelihood with a
hierarchical model"""

from __future__ import division
import numpy as np
import pymc3 as pm
from scipy.stats import norm
from scipy.interpolate import spline
import matplotlib.pyplot as plt
from plot_post import plot_post
import hpd as hpd1
from HtWtDataGenerator import *
#statistical data visualization package known as seaborn
import seaborn as sns

#simulation data using htwt generator
n_subj=30
HtWtData=HtWtDataGenerator(n_subj,rndsd=5678)
x=HtWtData[:,1]
y=HtWtData[:,2]
#recenter data at mean to reduce autocorr
#standardize to make initialization easier
x_m = np.mean(x)
x_sd = np.std(x)
y_m = np.mean(y)
y_sd = np.std(y)
zx = (x - x_m) / x_sd
zy = (y - y_m) / y_sd
#the model
with pm.Model() as model:
    #define the priors and then the likelihood
    tau = pm.Gamma('tau',0.001,0.001)
    beta0=pm.Normal('beta0', mu=0,tau=1.0E-12)
    beta1 = pm.Normal('beta1', mu=0, tau=1.0E-12)
    mu = beta0 + beta1 * zx
    #define likelihhood
    yl = pm.Normal('yl', mu=mu, tau=tau, observed=zy)
    #Generate the MCMC chain
    #you need to knwo the procedure of creating MCMC chain
    start1 = pm.find_MAP()
    step1 = pm.Metropolis()
    #you need to define init param which can be advi, advi_map, map, nuts
    # def sample(draws=500, step=None, init='auto', n_init=200000, start=None,
    #            trace=None, chain=0, njobs=1, tune=500, nuts_kwargs=None,
    #            step_kwargs=None, progressbar=True, model=None, random_seed=-1,
    #            live_plot=False, discard_tuned_samples=True, live_plot_kwargs=None,
    #            **kwargs):
    trace = pm.sample(10000, step=step1,init='auto', start=start1, progressbar=False)
#Examine the results
burnin=5000
thin=10

pm.traceplot(trace[burnin::thin])

## Extract chain values:
z0 = trace['beta0']
z1 = trace['beta1']
z_tau = trace['tau']
z_sigma = 1 / np.sqrt(z_tau) # Convert precision to SD


# Convert to original scale:
b1 = z1 * y_sd / x_sd
b0 = (z0 * y_sd + y_m - z1 * y_sd * x_m / x_sd)
sigma = z_sigma * y_sd


## Print summary for each trace
pm.summary(trace[burnin::thin])
pm.summary(trace)

# Posterior prediction:
# Specify x values for which predicted y's are needed:
x_post_pred = np.arange(55, 81)
# Define matrix for recording posterior predicted y values at each x value.
# One row per x value, with each row holding random predicted y values.
post_samp_size = len(b1)
#what does numpy zeroes - gives you an array or matrix of zeroes
y_post_pred = np.zeros((len(x_post_pred), post_samp_size))
# Define matrix for recording HDI limits of posterior predicted y values:
y_HDI_lim = np.zeros((len(x_post_pred), 2))
# Generate posterior predicted y values.
# This gets only one y value, at each x, for each step in the chain.
# what does this for mean
#     # what is connected to what meaning answer these
#     #y-distr ?
#     #x-distrf?
#     #beta 1 and beta 2 diest is what
#     #should check th e output here
#so this works as follows, it will predict value of y in each iteration of the MCMC sampkeing
#so if tehre are 20 loops, it will give 20 possible sample values of Y
for chain_idx in range(post_samp_size):
    y_post_pred[:,chain_idx] = norm.rvs(loc=b0[chain_idx] + b1[chain_idx] * x_post_pred ,
                           scale = np.repeat([sigma[chain_idx]], [len(x_post_pred)]), size=len(x_post_pred))

for x_idx in range(len(x_post_pred)):
    y_HDI_lim[x_idx] = pm.hpd(y_post_pred[x_idx])


## Display believable beta0 and b1 values
plt.figure()
plt.subplot(1, 2, 1)
thin_idx = 50
plt.plot(z1[::thin_idx], z0[::thin_idx], 'b.', alpha=0.7)
plt.ylabel('Standardized Intercept')
plt.xlabel('Standardized Slope')
plt.subplot(1, 2, 2)
plt.plot(b1[::thin_idx], b0[::thin_idx], 'b.', alpha=0.7)
plt.ylabel('Intercept (ht when wt=0)')
plt.xlabel('Slope (pounds per inch)')
plt.tight_layout()
plt.savefig('Figure_16.4.png')

# Display the posterior of the b1:
plt.figure(figsize=(8, 5))
plt.subplot(1, 2, 1)

# def plot_post(sample, alpha=0.05, show_mode=True, kde_plot=True, bins=50,
#     ROPE=None, comp_val=None, roundto=2)


plot_post(z1,comp_val=0.0, bins=30, show_mode=False)
plt.subplot(1, 2, 2)
plot_post(b1, comp_val=0.0, bins=30, show_mode=False)
plt.tight_layout()
plt.show()
