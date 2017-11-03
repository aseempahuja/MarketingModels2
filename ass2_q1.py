"""
Multiple linear regression
"""
from __future__ import division
import numpy as np
import pymc3 as pm
import pandas as pd
from scipy.stats import norm
import matplotlib.pyplot as plt
from plot_post import plot_post
from hpd import *
import seaborn as sns


# THE DATA.

fname = "ConjointAnalysis" # file name for saved graphs
data = pd.read_csv('clean_data2.csv', sep='\t')
# Specify variables to be used in BUGS analysis:
predictedName = "choice"
predictorNames = ["cargo3","engElec","engHyb","p35","p40","s7","s8","cp_d"]
nData = len(data)
y = data[predictedName].head(100)
x = data[predictorNames].head(100)
n_predictors = len(x.columns)

# THE MODEL
with pm.Model() as model:
    # define the priors
    beta0 = pm.Normal('beta0', mu=0, tau=1.0E-12)
    beta1 = pm.Normal('beta1', mu= 0, tau=1.0E-12, shape=n_predictors)
    tau = pm.Gamma('tau', 0.01, 0.01)

    mu=pm.Deterministic('mu',beta0+beta1[0]*x.values.T)
    # define the likelihood
    yl = pm.Normal('yl', mu=mu, tau=tau, observed=y)
    # Generate a MCMC chain
    start = pm.find_MAP()
    step1 = pm.NUTS([beta1])
    step2 = pm.Metropolis([beta0, tau])
    trace = pm.sample(10000, [step1, step2], start, progressbar=False)

# EXAMINE THE RESULTS
burnin = 5000
thin = 1

# Print summary for each trace
#pm.summary(trace[burnin::thin])
#pm.summary(trace)

# Check for mixing and autocorrelation
#pm.autocorrplot(trace[burnin::thin], vars =[mu, tau])
#pm.autocorrplot(trace, vars =[beta0])

## Plot KDE and sampled values for each parameter.
#pm.traceplot(trace[burnin::thin])
pm.traceplot(trace)

pm.summary(trace)

# # Extract chain values:
# b0_samp = trace['beta0'][burnin::thin]
# b_samp = trace['beta1'][burnin::thin]
# Tau_samp = trace['tau'][burnin::thin]
# Sigma_samp = 1 / np.sqrt(Tau_samp) # Convert precision to SD
# chain_length = len(Tau_samp)
# #
# # if n_predictors >= 6: # don't display if too many predictors
# #     n_predictors == 6
#
# columns = ['Sigma y', 'Intercept']
# [columns.append('Slope_%s' % i) for i in predictorNames[:n_predictors]]
# traces = np.array([Sigma_samp, b0_samp, b_samp[:,0], b_samp[:,1], b_samp[:,2], b_samp[:,3], b_samp[:,4], b_samp[:,5], b_samp[:,6], b_samp[:,7]]).T
# df = pd.DataFrame(traces, columns=columns)
# sns.set_style('dark')
# g = sns.PairGrid(df)
# g.map(plt.scatter)
# plt.savefig('Figure_17.5b.png')
#
# ## Display the posterior:
# sns.set_style('darkgrid')
#
# plt.figure(figsize=(16,4))
# plt.subplot(1, n_predictors+2, 1)
#
# # plot_post(z1,comp_val=0.0, bins=30, show_mode=False)
#
# plot_post(Sigma_samp, comp_val=0.0, bins=30, show_mode=False)
# plt.subplot(1, n_predictors+2, 2)
# plot_post(b0_samp,comp_val=0.0, bins=30, show_mode=False)
#
# for i in range(0, n_predictors):
#     plt.subplot(1, n_predictors+2, 3+i)
#     plot_post(b_samp[:,i], comp_val=0.0, bins=30, show_mode=False)
# plt.tight_layout()
# plt.savefig('Figure_17.5a.png')
#
#
# # Posterior prediction:
# # Define matrix for recording posterior predicted y values for each xPostPred.
# # One row per xPostPred value, with each row holding random predicted y values.
# y_post_pred = np.zeros((len(x), chain_length))
# # Define matrix for recording HDI limits of posterior predicted y values:
# y_HDI_lim = np.zeros((len(x), 2))
# # Generate posterior predicted y values.
# # This gets only one y value, at each x, for each step in the chain.
# #or chain_idx in range(chain_length):
# for chain_idx in range(chain_length):
#     y_post_pred[:,chain_idx] = norm.rvs(loc = b0_samp[chain_idx] + np.dot(b_samp[chain_idx], x.values.T),
#                                         scale = np.repeat([Sigma_samp[chain_idx]], [len(x)]))
#
# for x_idx in range(len(x)):
#     y_HDI_lim[x_idx] = pm.hpd(y_post_pred[x_idx])
#
# for i in range(len(x)):
#     print (np.mean(y_post_pred, axis=1)[i], y_HDI_lim[i])
#
# plt.show()