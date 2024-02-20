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
#from tqdm import tqdm




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
T = 300000
B = np.ones(n) / n
seed_count=20
#normalize v
v=(v-np.min(v))/(np.max(v)-np.min(v))
for i in range(n):
    for j in range(m):
        if(v[i][j]>1):
            v[i][j]=1



#
#
#
#
#calculate the optimal solution
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


#random 
result_random=[]
regret_random=[]

#calculate the regret
for i in range(seed_count):
    seed=i+1
    fpath = os.path.join('results', dataset, 'random_sd-{}'.format(seed))
    random_NSW= np.loadtxt(os.path.join(fpath, 'NSW.txt'))
    result_random.append(random_NSW)
    regret_random.append(abs(random_NSW-ENSW_all))

#calculate the average regret
regret_random_mean=np.zeros(T)
for t in range(T):
    for i in range(seed_count):
        regret_random_mean[t]+=regret_random[i][t]
regret_random_mean=regret_random_mean/seed_count



#fig = plt.figure(figsize=(6, 4))
#plt.plot(regret_random_mean,color='black')
#plt.title('random regret')
#plt.xlabel('steps')
#plt.ylabel('regret')
#plt.savefig(os.path.join('plots', 'random.pdf'))
#plt.clf()





#UCB
result_UCB=[]
regret_UCB=[]
#calculate the regret
for i in range(seed_count):
    seed=i+1
    fpath = os.path.join('results', dataset, 'UCB_greedy_sd-{}'.format(seed))
    UCB_NSW= np.loadtxt(os.path.join(fpath, 'NSW.txt'))
    result_UCB.append(UCB_NSW)
    regret_UCB.append(abs(UCB_NSW-ENSW_all))

#calculate the average regret
regret_UCB_mean=np.zeros(T)
for t in range(T):
    for i in range(seed_count):
        regret_UCB_mean[t]+=regret_UCB[i][t]
regret_UCB_mean=regret_UCB_mean/seed_count



#fig = plt.figure(figsize=(6, 4))
#plt.plot(regret_UCB_mean,color='purple')
#plt.title('UCB regret')
#plt.xlabel('steps')
#plt.ylabel('regret')
#plt.savefig(os.path.join('plots', 'UCB.pdf'))
#plt.clf()


#
#DA-greedy
result_greedy=[]
regret_greedy=[]
#calculate the regret
for i in range(seed_count):
    seed=i+1
    fpath = os.path.join('results', dataset, 'greedy_sd-{}'.format(seed))
    greedy_NSW= np.loadtxt(os.path.join(fpath, 'NSW.txt'))
    result_greedy.append(greedy_NSW)
    regret_greedy.append(abs(greedy_NSW-ENSW_all))

#calculate the average regret
regret_greedy_mean=np.zeros(T)
for t in range(T):
    for i in range(seed_count):
        regret_greedy_mean[t]+=regret_greedy[i][t]
regret_greedy_mean=regret_greedy_mean/seed_count



#fig = plt.figure(figsize=(6, 4))
#plt.plot(regret_greedy_mean,color='green')
#plt.title('DA-greedy regret')
#plt.xlabel('steps')
#plt.ylabel('regret')
#plt.savefig(os.path.join('plots', 'DA-greedy.pdf'))
#plt.clf()
#
#





#DA-EtC
result_ETC=[]
regret_ETC=[]
#calculate the regret
for i in range(seed_count):
    seed=i+1
    fpath = os.path.join('results', dataset, 'ETC_sd-{}'.format(seed))
    ETC_NSW= np.loadtxt(os.path.join(fpath, 'NSW.txt'))
    result_ETC.append(ETC_NSW)
    regret_ETC.append(abs(ETC_NSW-ENSW_all))

#calculate the average regret
regret_ETC_mean=np.zeros(T)
for t in range(T):
    for i in range(seed_count):
        regret_ETC_mean[t]+=regret_ETC[i][t]
regret_ETC_mean=regret_ETC_mean/seed_count



#fig = plt.figure(figsize=(6, 4))
#plt.plot(regret_ETC_mean,color='blue')
#plt.title('DA-EtC regret')
#plt.xlabel('steps')
#plt.ylabel('regret')
#plt.savefig(os.path.join('plots', 'DA-EtC.pdf'))
#plt.clf()


#DA-UCB
result_DA_UCB=[]
regret_DA_UCB=[]
#calculate the regret
for i in range(seed_count):
    seed=i+1
    fpath = os.path.join('results', dataset, 'UCB_sd-{}'.format(seed))
    DA_UCB_NSW= np.loadtxt(os.path.join(fpath, 'NSW.txt'))
    result_DA_UCB.append(DA_UCB_NSW)
    regret_DA_UCB.append(abs(DA_UCB_NSW-ENSW_all))

#calculate the average regret
regret_DA_UCB_mean=np.zeros(T)
for t in range(T):
    for i in range(seed_count):
        regret_DA_UCB_mean[t]+=regret_DA_UCB[i][t]
regret_DA_UCB_mean=regret_DA_UCB_mean/seed_count



#fig = plt.figure(figsize=(6, 4))
#plt.plot(regret_DA_UCB_mean,color='orange')
#plt.title('DA-UCB regret')
#plt.xlabel('steps')
#plt.ylabel('regret')
#plt.savefig(os.path.join('plots', 'DA-UCB.pdf'))
#plt.clf()

#save the results
fpath = os.path.join('results', dataset.lower(), 'NSW_result_mean')
os.makedirs(fpath, exist_ok=True)
np.savetxt(os.path.join(fpath, 'regret_random_mean.txt'), regret_random_mean, fmt='%.4e')
np.savetxt(os.path.join(fpath, 'regret_ETC_mean.txt'), regret_ETC_mean, fmt='%.4e')
np.savetxt(os.path.join(fpath, 'regret_greedy_mean.txt'), regret_greedy_mean, fmt='%.4e')
np.savetxt(os.path.join(fpath, 'regret_DA_UCB_mean.txt'), regret_DA_UCB_mean, fmt='%.4e')
np.savetxt(os.path.join(fpath, 'regret_UCB_mean.txt'), regret_UCB_mean, fmt='%.4e')

#plot the results
fig = plt.figure(figsize=(6, 4))
plt.plot(regret_random_mean,marker="o",markevery=0.1,label='Random',color='black',markersize=6)
plt.plot(regret_ETC_mean,marker="v",markevery=0.1,label='DA-ETC',color='blue',markersize=6)
plt.plot(regret_greedy_mean,marker="D",markevery=0.1,label='DA-greedy',color='green',markersize=6)
plt.plot(regret_DA_UCB_mean,marker=">",markevery=0.1,label='DA-UCB',color='orange',markersize=6)
plt.plot(regret_UCB_mean,marker="s",markevery=0.1,label='UCB',color='purple',markersize=6)
plt.title('Uniform', fontsize=18)
#plt.title('Jester', fontsize=18)
#plt.title('Household', fontsize=18)
plt.xlabel('rounds',fontsize=12)
plt.ylabel('regret',fontsize=12)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.grid(linestyle='dotted')
plt.legend()
plt.savefig(os.path.join('plots', 'ALL.pdf'))
plt.clf()


fig = plt.figure(figsize=(6, 4))
plt.plot(regret_ETC_mean,marker="v",markevery=0.1,label='DA-ETC',color='blue',markersize=6)
plt.plot(regret_greedy_mean,marker="D",markevery=0.1,label='DA-greedy',color='green',markersize=6)
plt.plot(regret_DA_UCB_mean,marker=">",markevery=0.1,label='DA-UCB',color='orange',markersize=6)
plt.title('Uniform', fontsize=18)
#plt.title('Jester', fontsize=18)
#plt.title('Household', fontsize=18)
plt.xlabel('rounds',fontsize=12)
plt.ylabel('regret',fontsize=12)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.grid(linestyle='dotted')
plt.legend()
plt.savefig(os.path.join('plots', 'DA-plot.pdf'))
plt.clf()

#fig = plt.figure(figsize=(6, 4))
#plt.plot(regret_ep_mean, label='DA-epsilon-greedy(ep=0.1)')
#plt.plot(regret_ETC_mean, label='DA-ETC',color='blue')
#plt.plot(regret_DA_UCB_mean, label='DA-UCB',color='orange')
#plt.title('Example2 regret', fontsize=18)
#plt.xlabel('rounds',fontsize=12)
#plt.ylabel('regret',fontsize=12)
#plt.legend()
#plt.savefig(os.path.join('plots', 'DA-ETC_UCB.pdf'))
#plt.clf()
