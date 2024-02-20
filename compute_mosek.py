import numpy as np
from numpy.core.records import _deprecate_shape_0_as_None
import pandas as pd
from utils import *
import argparse
#from sklearn.preprocessing import MinMaxScaler
import math


dataset = 'Uniform_10_10'
#dataset = 'Jokes_10_50'
#dataset = 'household_50_50'
#dataset = 'household_10_50'

# np.random.seed(1) # this seed is only for subsampling rows and columns
if dataset == 'Uniform_10_10':
    df = pd.read_csv("./dataset/10_10.csv",  header=None)
    v = df.to_numpy()
if dataset == 'Jokes_10_50':
    df = pd.read_csv("./dataset/Jokes_10_50.csv",  header=None)
    v = df.to_numpy()
if dataset == 'household_50_50':
    df = pd.read_csv("./dataset/household_50_50.csv",  header=None)
    v = df.to_numpy()
if dataset == 'household_10_50':
    df = pd.read_csv("./dataset/household_10_50.csv",  header=None)
    v = df.to_numpy()

# solve the instance (normalization done in the function)
n, m = v.shape



B = np.ones(n) / n
s=np.ones(m)/m

v=(v-np.min(v))/(np.max(v)-np.min(v))
#print(v)
x = mosek(v, B, s)

# save the result
b=compute_b_from_x(x,v,B)
p = np.sum(b, axis = 0)*m
import os
fpath = os.path.join('results', dataset.lower(), 'offline-eq_mosek')
os.makedirs(fpath, exist_ok=True)
np.savetxt(os.path.join(fpath, 'x'), x, fmt='%.4e')
np.savetxt(os.path.join(fpath, 'p'), x, fmt='%.4e')