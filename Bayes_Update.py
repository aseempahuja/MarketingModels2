"""
Bayesian updating of a coin, prior and posterior
distribution indiacte pmf at theta
"""
import matplotlib.pyplot as plt
import numpy as np

#How does prior and posterior work
#Posterior is what we are interested in
#Posterior propotional to Likelihood * prior
#Likelihood - observed data
#prior - choose something from past literature
#theta array of candidate values for the param theta
n_theta_vals = 3 #no. of candidate theta values
theta = np.linspace\
    (1/(n_theta_vals +1),
     n_theta_vals /(n_theta_vals +1),
     n_theta_vals )

#prior probabilities
# what is a triangular belief distribution
#pdf shaped like a triangle min - a, max - b
# and peak of c
p_theta=np.minimum(theta, 1-theta)
p_theta=p_theta/np.sum(p_theta)
##dgp
#what does numpy repeat function do
#to repeat elements fo an array
data = np.repeat([1, 0], [3, 9])
# print(data)
# [1 1 1 0 0 0 0 0 0 0 0 0]
n_heads = np.sum(data)
n_tails = len(data) - \
          n_heads
# Compute the likelihood of the data for each value of theta:
p_data_given_theta = theta**n_heads \
                     * (1-theta)**n_tails
# Compute the posterior:
p_data = np.sum(p_data_given_theta * p_theta)
p_theta_given_data = p_data_given_theta * \
                     p_theta / p_data
# This is Bayes' rule!
# Plot the results.
plt.figure(figsize=(12, 11))
plt.subplots_adjust(hspace=0.7)

# Plot the prior:
plt.subplot(3, 1, 1)
plt.stem(theta, p_theta, markerfmt=' ')
plt.xlim(0, 1)
plt.xlabel('$\\theta$')
plt.ylabel('$P(\\theta)$')
plt.title('Prior')
# Plot the likelihood:
plt.subplot(3, 1, 2)
plt.stem(theta, p_data_given_theta, markerfmt=' ')
plt.xlim(0, 1)
plt.xlabel('$\\theta$')
plt.ylabel('$P(D|\\theta)$')
plt.title('Likelihood')
plt.text(0.6, np.max(p_data_given_theta)/2, 'D = %sH,%sT' % (n_heads, n_tails))
# Plot the posterior:
plt.subplot(3, 1, 3)
plt.stem(theta, p_theta_given_data, markerfmt=' ')
plt.xlim(0, 1)
plt.xlabel('$\\theta$')
plt.ylabel('$P(\\theta|D)$')
plt.title('Posterior')
plt.text(0.6, np.max(p_theta_given_data)/2, 'P(D) = %g' % p_data)

plt.show()