from joblib import Parallel, delayed
import multiprocessing

inputs = range(10)

def processInput(i):
    result = 0
    for num in range(i):
        result += pow(num,num)
    return i,i+1

num_cores = multiprocessing.cpu_count()

results = Parallel(n_jobs=num_cores)(delayed(processInput)(i) for i in inputs)
print(results)

#%%
