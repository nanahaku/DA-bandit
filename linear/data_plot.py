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

plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

#set parameters
n=5
m=10
T =10000
#set the number of experiments
seed_count=10
B = np.ones(n) / n

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

n, m = v.shape
for i in range(n):
    for j in range(m):
        if(v[i][j]>1):
            v[i][j]=1




#offline optimal
fpath = os.path.join('results', dataset, 'offline-eq_mosek')
x_opt = np.loadtxt(os.path.join(fpath, 'x'))
#p_opt = np.loadtxt(os.path.join(fpath, 'p'))
u_opt = np.sum(v * x_opt, axis = 1)
#beta_opt = B / u_opt

#calculate the optimal NSW
ENSW=1
u_sum=np.zeros(n)
ENSW_all=np.zeros(T)
for i in range(n):
    for j in range(m):
        u_sum[i]+=x_opt[i][j]*v[i][j]
    u_sum[i]=pow(u_sum[i],B[i])
    ENSW=ENSW*u_sum[i]
for i in range(T):
        ENSW_all[i]=ENSW*(i+1)



#DA-EtC
result_naive_ETC=[]
regret_naive_ETC=[]
#calculate the regret
for i in range(seed_count):
    seed=i+1
    fpath = os.path.join('results', dataset, 'linear_ETC_naive_sd-{}'.format(seed))
    naive_NSW= np.loadtxt(os.path.join(fpath, 'NSW.txt'))
    result_naive_ETC.append(naive_NSW)
    regret_naive_ETC.append(abs(naive_NSW-ENSW_all))

#calculate the average regret
regret_naive_mean=np.zeros(T)
for t in range(T):
    for i in range(seed_count):
        regret_naive_mean[t]+=regret_naive_ETC[i][t]
regret_naive_mean=regret_naive_mean/seed_count



#Linear-DA-EtC
result_ridge_ETC=[]
regret_ridge_ETC=[]

#calculate the regret
for i in range(seed_count):
    seed=i+1
    fpath = os.path.join('results', dataset, 'linear_ETC_ridge_sd-{}'.format(seed))
    ridge_NSW= np.loadtxt(os.path.join(fpath, 'NSW.txt'))
    result_ridge_ETC.append(ridge_NSW)
    regret_ridge_ETC.append(abs(ridge_NSW-ENSW_all))

#calculate the average regret
regret_ridge_mean=np.zeros(T)
for t in range(T):
    for i in range(seed_count):
        regret_ridge_mean[t]+=regret_ridge_ETC[i][t]

regret_ridge_mean=regret_ridge_mean/seed_count



#linear-DA-UCB
result_linear_UCB=[]
regret_linear_UCB=[]
#calculate the regret
for i in range(seed_count):
    seed=i+1
    fpath = os.path.join('results', dataset, 'linear_UCB_alpha0.1_sd-{}'.format(seed))
    linear_NSW= np.loadtxt(os.path.join(fpath, 'NSW.txt'))
    result_linear_UCB.append(linear_NSW)
    #ENSW_allを50000まで省略
    ENSW_all1=ENSW_all[:50000]

    regret_linear_UCB.append(abs(linear_NSW-ENSW_all))

#calculate the average regret
regret_linear_UCB_mean=np.zeros(T)
for t in range(T):
    for i in range(seed_count):
        regret_linear_UCB_mean[t]+=regret_linear_UCB[i][t]
regret_linear_UCB_mean=regret_linear_UCB_mean/seed_count



#DA-UCB
result_DA_UCB=[]
regret_DA_UCB=[]
#calculate the regret
for i in range(seed_count):
    seed=i+1
    fpath = os.path.join('results', dataset, 'naive_UCB_sd-{}'.format(seed))
    DA_UCB_NSW= np.loadtxt(os.path.join(fpath, 'NSW.txt'))
    result_DA_UCB.append(DA_UCB_NSW)
    regret_DA_UCB.append(abs(DA_UCB_NSW-ENSW_all))

#calculate the average regret
regret_DA_UCB_mean=np.zeros(T)

for t in range(T):
    for i in range(seed_count):
        regret_DA_UCB_mean[t]+=regret_DA_UCB[i][t]
regret_DA_UCB_mean=regret_DA_UCB_mean/seed_count





#save the results
fpath = os.path.join('results', dataset.lower(), 'NSW_result_mean')
os.makedirs(fpath, exist_ok=True)
np.savetxt(os.path.join(fpath, 'regret_naive_mean.txt'), regret_naive_mean, fmt='%.4e')
np.savetxt(os.path.join(fpath, 'regret_ridge_mean.txt'), regret_ridge_mean, fmt='%.4e')
np.savetxt(os.path.join(fpath, 'regret_linear_UCB_mean.txt'), regret_linear_UCB_mean, fmt='%.4e')
np.savetxt(os.path.join(fpath, 'regret_DA_UCB_mean.txt'), regret_DA_UCB_mean, fmt='%.4e')


#plot the results
fig = plt.figure(figsize=(6, 4))

plt.plot(regret_naive_mean,marker="D",markevery=int(T/10),label='DA-EtC',color='red',markersize=6)
plt.plot(regret_ridge_mean,marker="s",markevery=int(T/10),label='Linear-EtC',color='purple',markersize=6)
plt.plot(regret_linear_UCB_mean,marker="^",markevery=int(T/10),label='Linear-UCB',color='orange',markersize=6)
plt.plot(regret_DA_UCB_mean,marker="x",markevery=int(T/10),label='DA-UCB',color='black',markersize=6)

plt.title('Movielens_{}_{}'.format(n,m), fontsize=18)
plt.xlabel('rounds',fontsize=14)
plt.ylabel('regret',fontsize=14)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.grid(linestyle='dotted')
plt.legend()
plt.savefig(os.path.join('plots', 'regret_{}.pdf'.format(dataset)))
plt.clf()