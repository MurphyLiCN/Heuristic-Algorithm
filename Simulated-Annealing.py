import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import test_function

def eval_function(x):
    y = test_function.f1(x)
    return y

def generate_candidate(x,sigma):
    c = x + np.random.normal(0,scale=sigma,size=len(x))
    return c

def accept_candidate(c,x,T):
    delta = eval_function(c) - eval_function(x)
    if delta <= 0:
        return True
    elif delta > 0:
        u = np.random.uniform()
        p = np.exp(- delta / T )
        if u < p :
            return True
        elif u >= p:
            return False

#%%
a = accept_candidate(1,2,3)
#%%
