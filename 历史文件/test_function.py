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


#%%
%reset
#%%
# 加载所需的库
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
#%%
def initialization(parameter):
    #产生随机的工作表
    joblist = pd.DataFrame(columns=['d_min','d_max','w_o','w_f','p','w','start','stop'],
                           dtype = int)
    #Trapezoidal Suitability Function，打分函数，列标题代表区块开始的时间点
    tsf = pd.DataFrame(0,
                       index=range(parameter['time_units']),
                       columns=range(parameter['n']))
    # 依次生成每一个任务的属性
    for i in range(parameter['n']):
        #时间段：w,d_min,d_max,d
        d_min = np.random.randint(parameter['d_min'],parameter['d_max']+1)
        d_max = np.random.randint(d_min,parameter['d_max']+1)
        w = np.random.randint(d_max,parameter['w_max']+1)
        #时间点：w_o,w_f,start,stop
        w_o = np.random.randint(0,parameter['time_units']-w)
        w_f = w_o + w
        #优先级：p
        p = np.random.randint(1,parameter['p_max']+1)
        # 更新joblist
        joblist.loc[i] = (d_min,d_max,w_o,w_f,p,w,0,0)

        # 更新打分函数
        s_i = tsf[i].copy()
        # 梯形4个顶点
        a = w_o
        b = int( w_o + np.ceil((w - d_max)/2) )
        c = b + d_max
        d = w_f
        #赋值1
        s_i[range(b,c)] = 1
        #赋值2
        if b > a:
            degree_1 = 1/(2*(b-a))
            n_1 = 1
            for j in range(a,b):
                s_i[j] = degree_1 * (2 * n_1 - 1)
                n_1 += 1
        #赋值3
        if d > c:
            degree_2 = 1/(2*(d-c))
            n_2 = 1
            for j in range(d-1,c-1,-1):
                s_i[j] = degree_2 * (2 * n_2 - 1)
                n_2 += 1
        #更新打分函数
        tsf[i] = s_i

    return tsf,joblist
#%%
def pd_sequencing(parameter,job_list,job,TSF):
    pd_job = job.copy()
    pd_jobindex = list(job_list['index_list'])
    pd_skip = list(job_list['skip'])
    pd_worst = list(job_list['worst'])
    #总时间轴,1代表位置可用
    pd_timetable = pd.Series(1,index=range(parameter['time_units']))
    #初始化总时间轴，因为有可能已经有任务指定了时间
    for i in range(parameter['n']):
        pd_timetable.loc[range(pd_job.loc[i,'start'],pd_job.loc[i,'stop'])]=0
    #任务时间轴,1代表该时间在该任务的窗口中
    empty_timetable = pd.Series(0,index=range(parameter['time_units']))
    #建立空工作表，按顺序存放被选择的工作的序号
    chosen_job_list = list()

    #对每一个工作进行判断
    for i in range(len(pd_jobindex)):

        if pd_skip[i]==0:
            continue

        job_index = pd_jobindex[i]
        job_info = pd_job.loc[job_index,]

        d_min = job_info.loc['d_min']
        d_max = job_info.loc['d_max']
        w_o = job_info.loc['w_o']
        w_f = job_info.loc['w_f']
        p = job_info.loc['p']
        w = job_info.loc['w']
        #下序列用来存放每一个开始时间，长度为d_min的工作带来的收益
        position_value = pd.Series(0,index=range(parameter['time_units']))
        #假设没有可供使用的位置
        at_least_one = False
        #对每一个位置进行测试
        for j in range(w-d_min+1):
            test_start = w_o+j
            test_stop = test_start+d_min
            test_job = empty_timetable.copy()
            test_job.loc[range(test_start,test_stop)]=1
            #判断位置是否可用
            if np.sum(test_job*pd_timetable) == d_min:
                at_least_one = True
                position_value.loc[test_start] = \
                    np.sum(TSF.loc[range(test_start,test_stop),job_index])
        #如果有至少一个可用位置，则选择最大的那个
        if at_least_one:
            # 判断最大最小位置
            if pd_worst[i]==1:
                position_value_bw = position_value.max()
            else:
                position_value_bw = \
                    pd.Series(list(set(position_value))).sort_values()[1]

            decide_start = \
                np.min(position_value.index[position_value == position_value_bw])
            decide_stop = decide_start+d_min
            #更新工作表
            pd_job.loc[job_index,'start'] = decide_start
            pd_job.loc[job_index,'stop'] = decide_stop
            #更新工作序号表
            chosen_job_list.append(job_index)
        else:
            decide_start = 0
            decide_stop = 0
        #刚刚被占用的位置不再可用
        pd_timetable.loc[range(decide_start,decide_stop)]=0

    return pd_job,chosen_job_list
#%%
def pd_timing(pd_chosen_jobindex,pd_job):
    test_pd_job = pd_job.copy()
    #对每一个工作进行左右延伸
    for i in range(len(pd_chosen_jobindex)):
        job_index = pd_chosen_jobindex[i]
        job_info = test_pd_job.loc[job_index,]

        d_min = job_info.loc['d_min']
        d_max = job_info.loc['d_max']
        w_o = job_info.loc['w_o']
        w_f = job_info.loc['w_f']
        p = job_info.loc['p']
        w = job_info.loc['w']
        start = job_info.loc['start']
        stop = job_info.loc['stop']
        #左右边界
        new_start = np.nanmax([test_pd_job.loc[test_pd_job.loc[:,'stop']<=start,'stop'].max(),
                               w_o,stop-d_max])
        new_stop = np.nanmin([test_pd_job.loc[test_pd_job.loc[:,'start']>=stop,'start'].min(),
                              w_f,new_start+d_max])
        #左右延伸
        test_pd_job.loc[job_index,'start'] = new_start
        test_pd_job.loc[job_index,'stop'] = new_stop
    return test_pd_job
#%%
def get_timetable(job,parameter):
    empty_timetable = pd.Series(-1,index=range(parameter['time_units']))
    for i in range(parameter['n']):
        empty_timetable.loc[range(job.loc[i,'start'],job.loc[i,'stop'])]=i
    return empty_timetable
#%%
def score(JOB,TSF,parameter):
    s = 0
    for i in range(parameter['n']):
        start = JOB.loc[i,'start']
        stop = JOB.loc[i,'stop']
        p = JOB.loc[i,'p']

        s_i = 0
        for j in range(start,stop):
            s_i += TSF.loc[j,i]

        s += s_i * p
    return s
#%%
def pd_get_score(f_parameter, job_list, f_job, f_tsf):
    f_job_after_sequencing, f_job_index_sequencing_chosen = \
        pd_sequencing(f_parameter, job_list, f_job,f_tsf)
    f_job_after_timing = \
        pd_timing(f_job_index_sequencing_chosen, f_job_after_sequencing)
    f_score = score(f_job_after_timing, f_tsf, f_parameter)
    return f_score
#%% md
## LA算法
#%%
def LA(job, f_job_index, f_parameter, f_tsf):
    f_job = job.copy()

    #总时间轴,1代表位置可用
    f_timetable = pd.Series(1, index=range(f_parameter['time_units']))
    #任务时间轴,1代表该时间在该任务的窗口中
    empty_timetable = pd.Series(0, index=range(f_parameter['time_units']))

    for i in range(f_parameter['n']):
        job_index = f_job_index[i]
        job_info = f_job.loc[job_index,]

        d_min = job_info.loc['d_min']
        d_max = job_info.loc['d_max']
        w_o = job_info.loc['w_o']
        w_f = job_info.loc['w_f']
        p = job_info.loc['p']
        w = job_info.loc['w']

        test_score = 0
        best_start = 0
        best_stop = 0

        for start in range(w_o,w_f-d_min+1):
            for d in range(d_min,min(d_max,w_f-start)+1):
                stop = start+d
                test_timetable = empty_timetable.copy()
                test_timetable.loc[range(start,stop)]=1

                if np.sum(test_timetable * f_timetable) == d:
                    test_LA_job = f_job.copy()
                    test_LA_job.loc[job_index,'start'] = start
                    test_LA_job.loc[job_index,'stop'] = stop
                    test_job_index_ordered_by_p = f_job_index[(i + 1):]

                    job_list = pd.DataFrame(columns=[],
                                            dtype = int)

                    job_list.insert(job_list.shape[1],'index_list',test_job_index_ordered_by_p)
                    job_list.insert(job_list.shape[1],'skip',[1]*len(test_job_index_ordered_by_p))
                    job_list.insert(job_list.shape[1],'worst',[1]*len(test_job_index_ordered_by_p))

                    new_test_score = \
                        pd_get_score(f_parameter, job_list, test_LA_job, f_tsf)

                    test_score = max(test_score,new_test_score)
                    if new_test_score == test_score:
                        best_start = start
                        best_stop = stop

        f_job.loc[job_index, 'start'] = best_start
        f_job.loc[job_index, 'stop'] = best_stop

        f_timetable.loc[range(best_start, best_stop)]=0

    return f_job
#%% md
## GA算法
#%%
def PMX_crossover(parent1, parent2):
    '''
    parent1 and parent2 are 1D np.array
    '''
    rng = np.random.default_rng()

    cutoff_1, cutoff_2 = np.sort(rng.choice(np.arange(len(parent1)+1), size=2, replace=False))

    def PMX_one_offspring(p1, p2):
        offspring = np.zeros(len(p1), dtype=p1.dtype)

        # Copy the mapping section (middle) from parent1
        offspring[cutoff_1:cutoff_2] = p1[cutoff_1:cutoff_2]

        # copy the rest from parent2 (provided it's not already there
        for i in np.concatenate([np.arange(0,cutoff_1), np.arange(cutoff_2,len(p1))]):
            candidate = p2[i]
            while candidate in p1[cutoff_1:cutoff_2]: # allows for several successive mappings
                #print(f"Candidate {candidate} not valid in position {i}") # DEBUGONLY
                candidate = p2[np.where(p1 == candidate)[0][0]]
            offspring[i] = candidate
        return offspring

    offspring1 = PMX_one_offspring(parent1, parent2)

    return offspring1

#%%
def binomial_crossover(parent1, parent2):
    if parent1 == 1:
        if parent2 == 1:
            son = np.random.binomial(1,0.9)
        else:
            son = np.random.binomial(1,0.5)

    if parent1 == 0:
        if parent2 == 0:
            son = np.random.binomial(1,0.1)
        else:
            son = np.random.binomial(1,0.5)

    return son
#%%
#parent1 = np.array([1,2,3,4,5,6,7,8,9])
#parent2 = np.array([5,4,6,7,2,1,3,9,8])
#son1 = PMX_crossover(parent1, parent2)
#%%
def GA(JOB,parameter,TSF,iteration):
    GA_all = pd.DataFrame(columns=['index_list','skip','worst','score'])
    GA_job = JOB.copy()

    test = list(range(40))

    for i in range(100):
        index_list = []
        while not sorted(index_list)==test:
            index_list = np.arange(40)
            np.random.shuffle(index_list)
        skip = np.random.binomial(1,0.5,40)
        worst = np.random.binomial(1,0.5,40)

        job_list = pd.DataFrame(columns=[],
                                dtype = int)

        job_list.insert(job_list.shape[1],'index_list',index_list)
        job_list.insert(job_list.shape[1],'skip',skip)
        job_list.insert(job_list.shape[1],'worst',worst)

        GA_score = pd_get_score(parameter,job_list,GA_job,TSF)

        row = {'index_list':index_list,
               'skip':skip,
               'worst':worst,
               'score':GA_score}

        GA_all.loc[i] = row

    for i in range(iteration):
        GA_all.sort_values(by='score',ascending = False,inplace=True)
        son1 = []

        while not sorted(son1)==test:
            parent1_index = np.random.randint(50)
            parent2_index = np.random.randint(100)
            parent1 = GA_all.loc[parent1_index,'index_list']
            parent2 = GA_all.loc[parent2_index,'index_list']

            son1 = PMX_crossover(np.array(parent1),np.array(parent2))

        p1_skip = GA_all.loc[parent1_index,'skip']
        p1_worst = GA_all.loc[parent1_index,'worst']
        p2_skip = GA_all.loc[parent2_index,'skip']
        p2_worst = GA_all.loc[parent2_index,'worst']

        son_skip = []
        son_worst = []
        for j in range(len(son1)):
            one_job_index = son1[j]
            parent1 = pd.Series(parent1)
            parent2 = pd.Series(parent2)
            p1_one_job_index = parent1.index[parent1 == one_job_index]
            p2_one_job_index = parent2.index[parent2 == one_job_index]
            son_skip.append(binomial_crossover(
                p1_skip[list(p1_one_job_index)[0]],
                p2_skip[list(p2_one_job_index)[0]]))
            son_worst.append(binomial_crossover(
                p1_worst[list(p1_one_job_index)[0]],
                p2_worst[list(p2_one_job_index)[0]]))

        job_list = pd.DataFrame(columns=[],
                                dtype = int)

        job_list.insert(job_list.shape[1],'index_list',son1)
        job_list.insert(job_list.shape[1],'skip',son_skip)
        job_list.insert(job_list.shape[1],'worst',son_worst)


        GA_new_score = pd_get_score(parameter,job_list,GA_job,TSF)
        n = 0
        new_job_list = job_list.copy()
        while sum(GA_all['score'] == GA_new_score)>0 and n < 2:
            index_test = []
            while not sorted(index_test)==test:
                new_job_list = job_list
                index1 = np.random.randint(40)
                index2 = np.random.randint(40)
                mutation1 = new_job_list.copy().loc[index1]
                mutation2 = new_job_list.copy().loc[index2]
                new_job_list.loc[index1] = mutation2
                new_job_list.loc[index2] = mutation1
                index_test = new_job_list['index_list']

            GA_new_score = pd_get_score(parameter,new_job_list,GA_job,TSF)
            n += 1

        row = {'index_list':list(new_job_list['index_list']),
               'skip':list(new_job_list['skip']),
               'worst':list(new_job_list['worst']),
               'score':GA_new_score}
        GA_all.iloc[99] = row

    GA_score =max(GA_all['score'])

    return GA_score
#%%

#%% md
## 一次运行

#%%
def once(i):
    np.random.seed(i)

    # 设置参数
    parameter = {'n': 40,
                 'time_units': 100,
                 'w_max': 25,
                 'd_min': 1,
                 'd_max': 25,
                 'p_max': 10}
    #初始化
    TSF,JOB = initialization(parameter)
    job_index_ordered_by_p = JOB.sort_values(by='p', ascending=False).index
    job_list = pd.DataFrame(columns=[],
                            dtype = int)

    job_list.insert(job_list.shape[1],'index_list',job_index_ordered_by_p)
    job_list.insert(job_list.shape[1],'skip',[1]*40)
    job_list.insert(job_list.shape[1],'worst',[1]*40)

    pd_job = JOB.copy()
    pd_score = pd_get_score(parameter,job_list,pd_job,TSF)

    LA_job = LA(JOB, job_index_ordered_by_p, parameter, TSF)
    LA_score = score(LA_job, TSF, parameter)

    iteration = 10000
    GA_score = GA(JOB,parameter,TSF,iteration)

    return pd_score,LA_score,GA_score
#%%

#%%
result = pd.DataFrame(columns=['PD','LA','GA'])
run_time = []
for i in range(50):
    print(i)
    time_start=time.time()
    result.loc[0] = once(i)
    time_end=time.time()
    run_time.append(time_end-time_start)
    print('time cost',time_end-time_start,'s')

#%%
np.random.seed(1)

# 设置参数
parameter = {'n': 40,
             'time_units': 100,
             'w_max': 25,
             'd_min': 1,
             'd_max': 25,
             'p_max': 10}
#初始化
TSF,JOB = initialization(parameter)
job_index_ordered_by_p = JOB.sort_values(by='p', ascending=False).index
job_list = pd.DataFrame(columns=[],
                        dtype = int)

job_list.insert(job_list.shape[1],'index_list',job_index_ordered_by_p)
job_list.insert(job_list.shape[1],'skip',[1]*40)
job_list.insert(job_list.shape[1],'worst',[1]*40)
#%%
time_start=time.time()

pd_job = JOB.copy()
pd_score = pd_get_score(parameter,job_list,pd_job,TSF)

time_end=time.time()
print('time cost',time_end-time_start,'s')
#%%
time_start=time.time()

LA_job = LA(JOB, job_index_ordered_by_p, parameter, TSF)
LA_score = score(LA_job, TSF, parameter)

time_end=time.time()
print('time cost',time_end-time_start,'s')
#%%
time_start=time.time()

iteration = 1000
GA_score = GA(JOB,parameter,TSF,iteration)

time_end=time.time()
print('time cost',time_end-time_start,'s')
#%%
from joblib import Parallel, delayed
import multiprocessing

#%%

num_cores = multiprocessing.cpu_count()

result_1W = Parallel(n_jobs=num_cores - 2) \
    (delayed(once)(i) for i in range(32))

print(result_1W)

#%%
import pickle


def save_variable(v, filename):
    f = open(filename, 'wb')
    pickle.dump(v, f)
    f.close()
    return filename


def load_variavle(filename):
    f = open(filename, 'rb')
    r = pickle.load(f)
    f.close()
    return r

#%%
filename = save_variable(result, '/Users/murphy/Documents/GitHub/Heuristic-Algorithm/result_32.txt')
#%%
filename_64 = save_variable(result_64, '/Users/murphy/Documents/GitHub/Heuristic-Algorithm/result_64.txt')
#%%
filename_1W = save_variable(result_1W, '/Users/murphy/Documents/GitHub/Heuristic-Algorithm/result_1W.txt')
#%%
#已经跑好保存的结果，循环套循环挺慢的
result_1 = load_variavle('/Users/murphy/Documents/GitHub/Heuristic-Algorithm/result_32.txt')
result_2 = load_variavle('/Users/murphy/Documents/GitHub/Heuristic-Algorithm/result_64.txt')
result_3 = load_variavle('/Users/murphy/Documents/GitHub/Heuristic-Algorithm/result_1W.txt')
#%%
plot_result = pd.concat([pd.DataFrame(result_1),pd.DataFrame(result_2)])

#%%
plot_result.columns = ['PD','LA','GA']
plot_result.index = range(64)

#%%
plot_result = pd.DataFrame(result_3)
plot_result.columns = ['PD','LA','GA']
#%%
plt.plot(plot_result)
#%%
plt.plot(plot_result[['PD','GA']])

#%%
plt.plot(plot_result[['LA','GA']])

#%%
plt.plot(plot_result['GA']-plot_result['LA'])
plt.hlines(0,0,32)
#%%

#%%
