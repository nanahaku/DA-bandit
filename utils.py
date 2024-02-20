
import numpy as np
import cvxpy as cp
np.random.seed(888)



def compute_b_from_x(x, v, B):
    tt = v*x
    return ((tt.T / np.sum(tt, 1)) * B).T

def eg_primal_obj_val(x, v, B):
    return np.sum(B * np.log(np.sum(v*x, 1))) # maximization obj

def eg_dual_obj_val(beta, v, B):
    m = v.shape[1]
    return np.sum(np.max(beta * v.T, 1))/m - np.sum(B * np.log(beta)) + np.sum(B * np.log(B)) - np.sum(B)



##################################### call Mosek directly ###################################
import scipy as sp
from scipy import sparse
import scipy.linalg as spla
from time import time
from mosek import *
from mosek.fusion import *
from scipy.sparse import coo_matrix, block_diag


def mosek(value, B, s = None):
    n, m = value.shape
    capacity = s
    x = cp.Variable((n,m),pos=True)  # Define x to represent the allocation (pos=True ensures it's non-negative)
    function=cp.sum(cp.multiply(B,cp.log(cp.diag(value@x.T))))
    # Each element of cp.diag(value@x.T) is the same as each player's utility u_i
    # By taking the log (base e) of that value and multiplying it by the budget, we represent the objective function of the EG problem
    objective = cp.Maximize(function)  # Maximize the function
    allocation=cp.cumsum(x,axis=0)  # By taking the cumulative sum of x by column, the value of the n-1 row (the bottom row) of allocation represents the second constraint condition of Σx in the figure above
    constraints = [0 <= x, allocation[n-1] <= capacity]
    prob = cp.Problem(objective, constraints)  # Define the problem
    result = prob.solve(solver=cp.MOSEK)  # Store the calculation result in result using SCS as the solver
    # Output the result below
    return x.value



