import matplotlib.pyplot as plot
import numpy as np

x=np.linspace(-20,20,4000)
# np.linspace() to create an array of equally spaced values.
# By declaring a start value, stop value, and the num of points in between those
# points an array will be generated.
y=np.sin(x)
plot.plot(x,y)
plot.savefig('fig2.2.png')
plot.show();