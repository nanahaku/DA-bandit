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
for k in range(10):#set the number of experiments
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

    NSW=1
    NSW_all=np.zeros(T)
    g=np.zeros(n)
    beta_all=[]
    beta_all.append(beta)
    u_all=[]
    u_all.append(np.zeros(n))
    for t in range(1, T+1):
        j = np.random.choice(m)
        items_all_t[t-1] = j
        has_budget = np.ones(n, dtype=bool)
        winner=random.randint(1,n)-1# Determine the winner rondomly
        count_sum[winner]+=1#count
        count[winner][j]+=1
        winners_all_t[t-1] = winner
        spending[winner] += beta[winner] * v_prediction[winner, j] 
        g_ave = (t-1) * g_ave / t if t > 1 else np.zeros(n)
        g_ave[winner] += v_prediction[winner, j] / t
        u_all.append(g_ave)
        for i in range(n):
            if g_ave[i]==0:
                g[i]=math.inf
            else:
                g[i]=B[i]/g_ave[i]
        beta = np.maximum((1-delta0) * B, np.minimum(1 + delta0, g))
        beta_all.append(beta)
        x_cumulative[winner, j] += 1
        if t % (int(T//5)) == 0:
            print('t = {}'.format(t))
        if(np.random.binomial(1,v[winner][j])):
            v_sum[winner][j]+=1
        v_prediction[winner][j]=v_sum[winner][j]/count[winner][j]
        NSW=1
        for i in range(n):
            NSW=NSW*pow(g_ave[i]*t,B[i])
        NSW_all[t-1]=NSW
    t2 = time.time()
    elapsed_time.append(t2-t1)
    #save results
    fpath = os.path.join('results', dataset.lower(), 'random_sd-'+str(seed)+'')
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