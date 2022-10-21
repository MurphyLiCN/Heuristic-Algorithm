# 作业
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def f1(x):
    y = np.sum(np.power(x,2))
    return y

def f2(x):
    y = 100 * np.power(np.power(x[0],2) - x[1] , 2) + np.power(1 - x[0] , 2)
    return y

def f3(x):
    y = np.sum(np.floor(x))
    return y

f4_p = pd.DataFrame([np.tile([-32,-16,0,16,32],5) , np.repeat([-32,-16,0,16,32],5)]).T
f4_j = pd.Series(range(1,26))

def f4(x,f4_p,f4_j):
    y = np.power(np.sum(1 / (np.sum(np.power(f4_p-x,6),axis=1) + f4_j)) + 0.002 , -1)
    return y
