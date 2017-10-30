import csv

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc3 as pm
import scipy
import scipy.stats as stats
import seaborn.apionly as sns
import statsmodels.api as sm
import theano.tensor as tt

data=pd.read_csv("BMK6107Data2.csv")
reg=np.polyfit(str(data['X']), str(data['Y']),1);
print(reg)