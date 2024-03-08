# PACE-bandit
Implementation of "Learning Fair Division from Bandit Feedback" (https://arxiv.org/abs/2311.09068)



# Getting Started
We are running this with Python 3.7.16. We cannot guarantee functionality with other versions.
Also, we use the mosek module to find the optimal solution. Please refer to (https://www.mosek.com/).

The Jester dataset we used is from (Goldberg et al., 2001) (http://www.ieor.berkeley.edu/~goldberg/pubs/eigentaste.pdf).

The Household dataset is cited from (Kroer et al., 2021) (https://arxiv.org/abs/1901.06230).

Please refer to this URL (https://github.com/alexpeys/market_datasets) for datasets used other than Uniform.

In creating the source code, we referred to (Gao et al.)(https://proceedings.neurips.cc/paper/2021/hash/e562cd9c0768d5464b64cf61da7fc6bb-Abstract.html).

# Running the experiments
At first, choose the dataset to use from the following candidates:

```
'Uniform_10_10'
'Jokes_10_50'
'household_50_50'
'household_10_50'
```

Next, run:
```
python compute_mosek.py  
```


Then, set the parameters and run the following:
```
python experiment_DA-ETC.py
python experiment_DA-greedy.py
python experiment_DA-UCB.py
python experiment_random.py
python experiment_UCB.py
```

Finally, run the following for data plotting:
```
python data_plot.py
```