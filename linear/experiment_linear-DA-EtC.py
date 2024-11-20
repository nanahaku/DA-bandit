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
from sklearn.linear_model import Ridge
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
    print('ridge seed = {},n={},m={}'.format(seed,n,m))
    random.seed(seed)
    np.random.seed(seed)
    # set parameters
    delta0 = 0.95
    beta = np.ones(n)
    beta_ave = np.zeros(n)
    g_ave = np.zeros(n)
    items_all_t = np.zeros(T, dtype=int) # j(t) sampled uniformly at random from {0, 1, ..., m-1}
    winners_all_t = np.zeros(T, dtype=int) # i(t) = min of argmax over i of beta[i] * v[i, j(t)]
    spending = np.zeros(n)
    x_cumulative = np.zeros((n, m))
    x_proportional = (B * np.ones(shape=(n, m)).T).T / m
    v_prediction=np.zeros((n,m))
    H1_prediction=np.zeros((c,m))#parameter to be estimated
    
    
    
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
    


    NSW=1
    NSW_all=np.zeros(T)
    g=np.zeros(n)
    ETC=int(pow(T,2/3)*pow(n*m,1/3))# This value is arbitrary

    beta_all=[]
    beta_all.append(beta)# Store beta at t=0 in beta_all
    u_all=[]
    u_all.append(np.zeros(n))# Store \bar{u} at t=0 in u_all
    U_j=[]
    for t in range(1, ETC+1):
        j = np.random.choice(m)
        items_all_t[t-1] = j
        count_item[j]+=1
        winner=random.randint(1,n)-1 # Determine the winner
        X[j].append(W1[winner,:].tolist())
        count_sum[winner]+=1#count
        count[winner][j]+=1
        if(np.random.binomial(1,v[winner][j])):
            u_sum[winner][j]+=1
            u_sum_all[winner]+=1
            U[j].append(1)
        else:
            U[j].append(0)
        if t % (int(T//5)) == 0:# Visualization of progress
            print('t = {}'.format(t))
        NSW=1
        for i in range(n):
            NSW=NSW*pow(u_sum_all[i],B[i])
        NSW_all[t-1]=NSW
    for j in range(m): # Estimate the parameter H1
        N=count_item[j]
        X_j=np.array(X[j])
        U_j=np.array(U[j])
        ridge_model = Ridge(alpha=1.0, fit_intercept=False)
        ridge_model.fit(X_j, U_j)
        theta_j = ridge_model.coef_

        H1_prediction[:, j] = theta_j
    v_prediction=np.dot(W1, H1_prediction)
    for i in range(n):
        for j in range(m):
            if(v_prediction[i][j]>1):
                v_prediction[i][j]=1
    for t in range(ETC+1, T+1):# After ETC, run PACE
        t_eff = t - ETC 
        T_eff = T - ETC
        # sample an item
        j = np.random.choice(m)
        items_all_t[t-1] = j
        # remove buyers that have depleted their budgets
        has_budget = np.ones(n, dtype=bool)
        
        winner= random.choice(np.where(beta[has_budget] * v_prediction[has_budget, j] == np.max(beta[has_budget] * v_prediction[has_budget, j]))[0])
        #winner = np.argmax(beta[has_budget] * v_prediction[has_budget, j])
        count_sum[winner]+=1#count
        count[winner][j]+=1
        # find winners for this item (just pick the lex. smallest winner, if tie)
        winners_all_t[t-1] = winner
        v_noise=np.random.binomial(1,max(0,v[winner][j]))
        u_sum[winner][j]+=v_noise
        u_sum_all[winner]+=v_noise
        
        # update u
        g_ave = (t_eff-1) * g_ave / t_eff if t_eff > 1 else np.zeros(n)
        # note the m: since it is non-averaged sum over j
        g_ave[winner] += v_noise / t_eff
        u_all.append(g_ave)# Store \bar{u} at t in u_all
        # update beta
        for i in range(n):
            if g_ave[i]==0:
                g[i]=math.inf
            else:
                g[i]=B[i]/g_ave[i]
        beta = np.maximum((1-delta0) * B, np.minimum(1 + delta0, g))
        beta_ave = (t_eff-1) * beta_ave / t_eff + beta / t_eff
        beta_all.append(beta)
        
        x_cumulative[winner, j] += 1
        if t % (int(T//5)) == 0:# Visualization of progress
            print('t = {}'.format(t))

        NSW=1
        for i in range(n):
            NSW=NSW*pow(u_sum_all[i],B[i])
        NSW_all[t-1]=NSW
    t2 = time.time()
    elapsed_time.append(t2-t1)# Store the elapsed time
    fpath = os.path.join('results', dataset.lower(), 'linear_ETC_ridge_sd-'+str(seed)+'')
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