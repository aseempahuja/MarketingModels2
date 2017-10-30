#density fn can have values larger than 1
import matplotlib.pyplot as plt
import numpy as np

meanval=0.0
sdval=0.2
xlow= meanval-3*sdval
xhigh= meanval+3*sdval
dx=0.002 #interval width
#specify points between low and high points
# We can use np.arange() to create an array of values starting from the start value
# and incrementally going up to end value by incrementing up by the step value.
# step is by default set to 1. It is very similar to np.linspace as both output arrays
# which start and stop at given values and with a certain number of values in the array, or rather,
# with a step size that gets us from the start to the end.
x = np.arange(xlow, xhigh, dx)
# Compute y values, i.e., probability density at each value of x:
#y is also array of values
y=(1/(sdval*np.sqrt(2*np.pi)))*np.exp(-0.5*((x-meanval)/sdval)**2)
plt.plot(x,y)
# A stem plot plots vertical lines (using linefmt) at each x location from the baseline to y,
# and places a marker there using markerfmt.
# A horizontal line at 0 is plotted using basefmt.
plt.stem(x, y, markerfmt=' ')
print('hi')
plt.xlabel('$x$')
plt.ylabel('$p(x)$')
plt.title('Normal Probability Density')
# Approximate the integral as the sum of width * height for each interval.
area = np.sum(dx*y)
# print(area)
# Display info in the graph.
plt.text(-.6, 1.7, '$\mu$ = %s' % meanval)
plt.text(-.6, 1.5, '$\sigma$ = %s' % sdval)
plt.text(.2, 1.7, '$\Delta x$ = %s' % dx)
plt.text(.2, 1.5, '$\sum_{x}$ $\Delta x$ $p(x)$ = %5.3f' % area)
plt.show()
plt.savefig('normal.3.png')
