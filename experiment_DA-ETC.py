import numpy as np
import pandas as pd
from utils import *
import argparse, os
import random
import math
from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd
import json
import time
#from sklearn.preprocessing import MinMaxScaler


#select dataset
dataset = 'Uniform_10_10'
#dataset = 'Jokes_10_50'
#dataset = 'household_50_50'
#dataset = 'household_10_50'

#load dataset
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


#set parameters
n, m = v.shape
T = 100000
B = np.ones(n) / n
#normalize v
v=(v-np.min(v))/(np.max(v)-np.min(v))
for i in range(n):
    for j in range(m):
        if(v[i][j]>1):
            v[i][j]=1



#run the experiment
for k in range(20):# Determine the number of experiments
    elapsed_time=[]
    t1 = time.time()
    seed=k+1# Change the seed value for each experiment
    print('seed = {}'.format(seed))
    random.seed(seed)
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
    
    for i in range(n):
        for j in range(m):
            v_prediction[i][j]=1
    count_sum=np.zeros(n)
    count=np.zeros((n,m))
    v_sum = np.zeros((n, m))
    v_sum_all = np.zeros(n)


    NSW=1
    NSW_all=np.zeros(T)
    g=np.zeros(n)
    ETC=int(pow(T,2/3)*pow(n*m,1/3))# This value is arbitrary

    beta_all=[]
    beta_all.append(beta)# Store beta at t=0 in beta_all
    u_all=[]
    u_all.append(np.zeros(n))# Store \bar{u} at t=0 in u_all
    for t in range(1, ETC+1):
        j = np.random.choice(m)
        items_all_t[t-1] = j
        winner=random.randint(1,n)-1 # Determine the winner
        count_sum[winner]+=1#count
        count[winner][j]+=1
        if(np.random.binomial(1,v[winner][j])):
            v_sum[winner][j]+=1
            v_sum_all[winner]+=1
        if t % (int(T//5)) == 0:# Visualization of progress
            print('t = {}'.format(t))
        NSW=1
        for i in range(n):
            NSW=NSW*pow(v_sum_all[i],B[i])
        NSW_all[t-1]=NSW
    for i in range(n):
        for j in range(m):
            if(count[i][j]!=0):
                v_prediction[i][j]=v_sum[i][j]/count[i][j]
            else:
                v_prediction[i][j]=1

    for t in range(ETC+1, T+1):# After ETC, run PACE
        t_eff = t - ETC 
        T_eff = T - ETC
        # sample an item
        j = np.random.choice(m)
        items_all_t[t-1] = j
        # remove buyers that have depleted their budgets
        has_budget = np.ones(n, dtype=bool)
        winner = np.argmax(beta[has_budget] * v_prediction[has_budget, j])
        count_sum[winner]+=1#count
        count[winner][j]+=1
        # find winners for this item (just pick the lex. smallest winner, if tie)
        winners_all_t[t-1] = winner
        v_noise=np.random.binomial(1,v[winner][j])
        v_sum[winner][j]+=v_noise
        v_sum_all[winner]+=v_noise
        
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
            NSW=NSW*pow(v_sum_all[i],B[i])
        NSW_all[t-1]=NSW
    t2 = time.time()
    elapsed_time.append(t2-t1)# Store the elapsed time
    fpath = os.path.join('results', dataset.lower(), 'ETC_sd-'+str(seed)+'')
    # Save the results
    os.makedirs(fpath, exist_ok=True)
    np.savetxt(os.path.join(fpath, 'count.txt'), count, fmt='%.4e')
    np.savetxt(os.path.join(fpath, 'v_prediction.txt'), v_prediction, fmt='%.4e')
    np.savetxt(os.path.join(fpath, 'v_sum.txt'), v_sum, fmt='%.4e')
    np.savetxt(os.path.join(fpath, 'NSW.txt'), NSW_all, fmt='%.4e')
    np.savetxt(os.path.join(fpath, 'time.txt'), elapsed_time, fmt='%.4e')
    meta_data = {'T': T, 'dataset': dataset, 'n': n, 'm': m,  
                'seed': seed, 'delta0': delta0}
    with open(os.path.join(fpath, 'meta_data'), 'w') as mdff:
        mdff.write(json.dumps(meta_data, indent=4))