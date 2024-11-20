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
from scipy.optimize import nnls
#from sklearn.preprocessing import MinMaxScaler

start = time.time()
#set parameters
n=5
m=10
T = 10000
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

c=5 # Number of components

nmf = NMF(n_components=c)
nmf.fit(v)
W1 = nmf.fit_transform(v)
H1 = nmf.components_
v=np.dot(W1, H1)

n, m = v.shape
for i in range(n):
    for j in range(m):
        if(v[i][j]>1):
            v[i][j]=1
            




for k in range(10):# Determine the number of experiments
    elapsed_time=[]
    t1 = time.time()
    seed=k+1# Change the seed value for each experiment
    print('ucb seed = {}'.format(seed))
    np.random.seed(seed)
    random.seed(seed)
    # set parameters
    delta0 = 0.95
    beta = np.ones(n)
    beta_ave = np.zeros(n)
    g_ave = np.zeros(n)
    items_all_t = np.zeros(T, dtype=int) # j(t) sampled uniformly at random from {0, 1, ..., m-1}
    winners_all_t = np.zeros(T, dtype=int) # i(t) = min of argmax over i of beta[i] * v[i, j(t)]
    x_cumulative = np.zeros((n, m))
    x_proportional = (B * np.ones(shape=(n, m)).T).T / m
    v_prediction=np.zeros((n,m))
    H1_prediction=np.zeros((c,m))
    ucb_lambda=1.0
    UCB=np.ones((n,m))
    UCB_norm=np.zeros((n,m))
    spending = np.zeros(n)
    for i in range(n):
        for j in range(m):
                UCB[i][j]=1
    
    
    for i in range(c):
        for j in range(m):
            H1_prediction[i][j]=1
    
    count_sum=np.zeros(n)
    count_item=np.zeros(m)
    count=np.zeros((n,m))
    u_sum = np.zeros((n, m))
    u_sum_all = np.zeros(n)
    U = []
    X= []
    for j in range(m):
        U.append([])
        X.append([])
    CI=0
    


    NSW=1
    NSW_all=np.zeros(T)
    g=np.zeros(n)
    #ETC=int(pow(T,2/3)*pow(n*m,1/3))# This value is arbitrary
    #ETC=10000

    beta_all=[]
    beta_all.append(beta)# Store beta at t=0 in beta_all
    u_all=[]
    u_all.append(np.zeros(n))# Store \bar{u} at t=0 in u_all
    U_j=[]

    alpha_precomputed = np.sqrt(np.log(np.arange(1, T + 1)))


    for t in range(1, T+1):
        # sample an item
        j = np.random.choice(m)
        items_all_t[t-1] = j
        has_budget = np.ones(n, dtype=bool)
        # compute UCB value
        for i in range(n):
            if(count[i][j]==0):
                CI=1
            else:
                CI= math.sqrt(np.log(t)/(2 * count[i][j]))
            UCB[i][j]=np.minimum(1,v_prediction[i][j]+CI)
        UCB_norm = UCB # m * (UCB.T / np.sum(UCB, 1)).
        #print(beta)
        winner = random.choice(np.where(beta[has_budget] * UCB_norm[has_budget, j] == np.max(beta[has_budget] * UCB_norm[has_budget, j]))[0])
        #winner = np.argmax(beta[has_budget] * UCB_norm[has_budget, j])
        count_sum[winner]+=1#count
        count[winner][j]+=1
        # find winners for this item (just pick the lex. smallest winner, if tie)
        winners_all_t[t-1] = winner
        spending[winner] += beta[winner] * v_prediction[winner, j] 
        # update beta_ave
        g_ave = (t-1) * g_ave / t if t > 1 else np.zeros(n)
        g_ave[winner] += v_prediction[winner, j] / t
        u_all.append(g_ave)
        # update beta
        # update beta
        for i in range(n):
            if g_ave[i]==0:
                g[i]=math.inf
            else:
                g[i]=B[i]/g_ave[i]
        beta = np.maximum((1-delta0) * B, np.minimum(1 + delta0, g))
        beta_all.append(beta)#store beta at t+1 in beta_all
        x_cumulative[winner, j] += 1
        if t % (int(T//5)) == 0:# Visualization of progress
            print('t = {}'.format(t))
        if(np.random.binomial(1,v[winner][j])):
            u_sum[winner][j]+=1
            u_sum_all[winner]+=1
        v_prediction[winner][j]=u_sum[winner][j]/count[winner][j]
        NSW=1
        for i in range(n):
            NSW=NSW*pow(u_sum_all[i],B[i])
        #print(NSW)
        NSW_all[t-1]=NSW

    t2 = time.time()
    elapsed_time.append(t2-t1)# Store the elapsed time
    fpath = os.path.join('results', dataset.lower(), 'naive_UCB_sd-'+str(seed)+'')
    # Save the results
    os.makedirs(fpath, exist_ok=True)
    np.savetxt(os.path.join(fpath, 'count.txt'), count, fmt='%.4e')
    np.savetxt(os.path.join(fpath, 'v_prediction.txt'), v_prediction, fmt='%.4e')
    np.savetxt(os.path.join(fpath, 'NSW.txt'), NSW_all, fmt='%.4e')
    np.savetxt(os.path.join(fpath, 'time.txt'), elapsed_time, fmt='%.4e')
    np.savetxt(os.path.join(fpath, 'H1_prediction.txt'), H1_prediction, fmt='%.4e')
    np.savetxt(os.path.join(fpath, 'H1.txt'), H1, fmt='%.4e')
    np.savetxt(os.path.join(fpath, 'v.txt'), v, fmt='%.4e')
    meta_data = {'T': T, 'dataset': dataset, 'n': n, 'm': m,  
                'seed': seed, 'delta0': delta0}
    with open(os.path.join(fpath, 'meta_data'), 'w') as mdff:
        mdff.write(json.dumps(meta_data, indent=4))

end = time.time()  
time_diff = end - start  
time_diff = round(time_diff, 2)
print("time_diff",time_diff)