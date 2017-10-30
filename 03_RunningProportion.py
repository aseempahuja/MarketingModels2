# Goal: toss a coin n times and compute running proportionds of heads
import matplotlib.pyplot as plt
import numpy as np

#totla no. of flips
N= 5000
#generate random sample of n flips
np.random.seed()
#what is tghe signature?
flip_seq=np.random.choice(a=(0, 1), p=(.5, .5), size=N, replace=True)
#compute the running proportion of heads
#why do you calculate cumulative sum
#it tells us how much proportion of the total toss
#are heads cumulative sum/the index of the toss
r=np.cumsum(flip_seq)
#the x axis
n=np.linspace(1,N,N)
run_prop=r/n
#graph plotting '-o' the color of the line
plt.plot(n,run_prop,'-o',)
plt.xscale('log')
plt.xlim(1,N)
plt.ylim(0,1)
plt.axhline(y=.5, ls='dashed')
plt.show()

