"""
http://nbviewer.jupyter.org/github/aflaxman/pymc-examples/blob/master/seeds_re_logistic_regression_pymc3.ipynb
Read this also
https://dsaber.com/2016/08/27/analyze-your-experiment-with-a-multilevel-logistic-regression-using-pymc3​/
"""

import numpy as np, pymc3 as pm, theano.tensor as T
### data - same as PyMC2 version
# germinated seeds
r =  np.array([10, 23, 23, 26, 17, 5, 53, 55, 32, 46, 10, 8, 10, 8, 23, 0, 3, 22, 15, 32, 3])

# total seeds
n =  np.array([39, 62, 81, 51, 39, 6, 74, 72, 51, 79, 13, 16, 30, 28, 45, 4, 12, 41, 30, 51, 7])

# seed type
x1 = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])

# root type
x2 = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1])

# number of plates
N =  x1.shape[0]
### model - major differences from PyMC2:
###   * with model idiom
###   * initial values are not specified when constructing stochastics
###   * keyword `shape` instead of `size`
###   * Theano tensors instead of Lambda deterministics
###   * `observed` parameter takes a value instead of a boolean

with pm.Model() as m:
    ### hyperpriors
    tau = pm.Gamma('tau', 1.e-3, 1.e-3)
    sigma = tau**-.5

    ### parameters
    # fixed effects
    alpha_0 = pm.Normal('alpha_0', 0., 1e-6)
    alpha_1 = pm.Normal('alpha_1', 0., 1e-6)
    alpha_2 = pm.Normal('alpha_2', 0., 1e-6)
    alpha_12 = pm.Normal('alpha_12', 0., 1e-6)

    # random effect
    b = pm.Normal('b', 0., tau,  shape=(N,))

    # expected parameter
    logit_p =  (alpha_0 + alpha_1*x1 + alpha_2*x2 + alpha_12*x1*x2 + b)
    p = T.exp(logit_p) / (1 + T.exp(logit_p))

    ### likelihood
    obs = pm.Binomial('obs', n, p, observed=r)
    # %%time

### inference - major differences from PyMC2:
###   * find_MAP function instead of MAP object
###   * initial values specified in inference functions
###   * selection of MCMC step methods is more explicit
###   * sampling from parallel chains is super-simple

n = 2000

with m:
    start = pm.find_MAP({'tau': 10., 'alpha_0': 0., 'alpha_1': 0., 'alpha_2': 0., 'alpha_12': 0., 'b': np.zeros(N)})
    step = pm.HamiltonianMC(scaling=start)

    ptrace = pm.psample(n, step, start, progressbar=False, threads=4)
    ### inference - major differences from PyMC2:
###   * find_MAP function instead of MAP object
###   * initial values specified in inference functions
###   * selection of MCMC step methods is more explicit
###   * sampling from parallel chains is super-simple

n = 2000

with m:
    start = pm.find_MAP({'tau': 10., 'alpha_0': 0., 'alpha_1': 0., 'alpha_2': 0., 'alpha_12': 0., 'b': np.zeros(N)})
    step = pm.HamiltonianMC(scaling=start)

    ptrace = pm.psample(n, step, start, progressbar=False, threads=4)

burn = 1000
pm.summary(ptrace[burn:])
pm.gelman_rubin(ptrace[burn:])
