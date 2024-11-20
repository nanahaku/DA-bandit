import numpy as np
from sklearn.decomposition import NMF
from numpy.core.records import _deprecate_shape_0_as_None
import pandas as pd
from utils import *
#from sklearn.preprocessing import MinMaxScaler
import math
import argparse, os
import random
from matplotlib import pyplot as plt
import seaborn as sns
import json
import time

#set parameters
n=5
m=10

#select dataset
df = pd.read_csv("./dataset/movielens_1500_1500.csv",  header=None)
dataset='movielens_{}_{}'.format(n,m)

np.random.seed(1)
v = df.to_numpy()
#select n×m submatrix from v
v = v[np.random.choice(v.shape[0], n, replace=False)]
v = v[:, np.random.choice(v.shape[1], m, replace=False)]

#normalize v
v=(v-np.min(v))/(np.max(v)-np.min(v))




#set components
nmf = NMF(n_components=5)
nmf.fit(v)
W1 = nmf.fit_transform(v)
H1 = nmf.components_
v=np.dot(W1, H1)
#if v[i][j]>1, v[i][j]=1
for i in range(n):
    for j in range(m):
        if(v[i][j]>1):
            v[i][j]=1

B = np.ones(n) / n
s=np.ones(m)/m

# solve the problem
x = mosek(v, B, s)

# save the result
b=compute_b_from_x(x,v,B)
p = np.sum(b, axis = 0)*m
import os
fpath = os.path.join('results', dataset.lower(), 'offline-eq_mosek')
os.makedirs(fpath, exist_ok=True)
np.savetxt(os.path.join(fpath, 'x'), x, fmt='%.4e')
np.savetxt(os.path.join(fpath, 'p'), x, fmt='%.4e')