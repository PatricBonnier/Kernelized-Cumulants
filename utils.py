import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from math import factorial
from scipy.spatial.distance import squareform, pdist, cdist
from scipy.stats import norm
import pandas as pd

BANDWIDTHS      = [0.00001, 0.000025,  0.00005,    0.000075, 
                   0.0001,  0.00025,   0.0005,     0.00075, 
                   0.001,   0.0025,    0.005,      0.0075, 
                   0.01,    0.025,     0.05,       0.075, 
                   0.1,     0.25,      0.5,        0.75, 
                   1.0,     2.5,       5.0,        7.5]

#### ---- Permutation utils

def two_sample_permutation_test(test_statistic, X, Y, num_permutations, prog_bar=True):
    assert X.ndim == Y.ndim
    statistics = np.zeros(num_permutations)
    range_ = range(num_permutations)
    if prog_bar:
        range_ = tqdm(range_)
    for i in range_:
        if X.ndim == 1:
            Z = np.hstack((X,Y))
        elif X.ndim == 2:
            Z = np.vstack((X,Y))
            
        perm_inds = np.random.permutation(len(Z))
        Z = Z[perm_inds]
        X_ = Z[:len(X)]
        Y_ = Z[len(X):]
        my_test_statistic = test_statistic(X_, Y_)
        statistics[i] = my_test_statistic
    return statistics

def gauss_kernel(X, Y=None, sigma=1.0):
    if Y is None:
        sq_dists = squareform(pdist(X, 'sqeuclidean'))
    else:
        sq_dists = cdist(X, Y, 'sqeuclidean')
    K = np.exp(-sq_dists / (2 * sigma**2))
    return K

def KME(X,Y,kernel):
    assert X.ndim == Y.ndim == 2
    K_XX = kernel(X, X)
    K_XY = kernel(X, Y)
    K_YY = kernel(Y, Y)
       
    n = len(K_XX)
    m = len(K_YY)
    
    np.fill_diagonal(K_XX, 0)
    np.fill_diagonal(K_YY, 0)
    mmd = np.sum(K_XX) / (n*(n-1))  + np.sum(K_YY) / (m*(m-1))  - 2 * np.sum(K_XY)/(n*m)
    return mmd

def KVE(X,Y,kernel):
    assert X.ndim == Y.ndim == 2
    K_XX = kernel(X, X)
    K_XY = kernel(X, Y)
    K_YY = kernel(Y, Y)
       
    n = len(X)
    m = len(Y)
    
    K_XX -= K_XX.sum(axis=1, keepdims=True) / n
    K_YY -= K_YY.sum(axis=1, keepdims=True) / m
    K_YXt  = K_XY - K_XY.sum(axis=0, keepdims=True) / n
    K_XY -= K_XY.sum(axis=1, keepdims=True) / m
    
    mmd = np.sum(K_XX * K_XX.transpose()) / (n*n) + np.sum(K_YY * K_YY.transpose()) / (m*m) - 2.0 * np.sum(K_XY * K_YXt) / (n*m)
    return mmd

def quadratic_time_mmd_var(X,Y,kernel): 
    assert X.ndim == Y.ndim == 2
    K_XX = kernel(X, X)
    K_XY = kernel(X,Y)
    K_YX = kernel(Y,X)
    K_YY = kernel(Y, Y)
       
    n = len(K_XX)
    m = len(K_YY)
    
    Hn = np.identity(n) - np.ones((n,n))/n
    Hm = np.identity(m) - np.ones((m,m))/m
    K_XX = np.matmul(K_XX, Hn)
    K_YY = np.matmul(K_YY, Hm)
    K_XY = np.matmul(K_XY, Hm)
    K_YX = np.matmul(K_YX, Hn)
    
    mmd = np.trace(np.matmul(K_XX,K_XX)) / (n*n) \
        + np.trace(np.matmul(K_YY, K_YY)) / (m*m) \
    - 2 * np.trace(np.matmul(K_XY, K_YX)) / (n*m)
    return mmd

def HSIC(X,Y,k,l):
    K = k(X,X)
    L = l(Y,Y)
    n = len(K)
    H = np.identity(n) - np.ones((n,n))/n
    return np.trace(np.matmul(np.matmul(K,H),np.matmul(L,H)))/(n*n)

def CSIC(X,Y,k,l): 
    K = k(X,X)
    L = l(Y,Y)
    n = len(K)

    Kr = K.sum(axis=1, keepdims=True) / n
    Lr = L.sum(axis=1, keepdims=True) / n
    Kl = K.sum(axis=0, keepdims=True) / n
    Ll = L.sum(axis=0, keepdims=True) / n
    
    KK = np.sum(K)/(n*n)
    LL = np.sum(L)/(n*n)
    
    return np.sum( K*K*L   - 4*K*Kr*L  - 2*K*K*Lr  + 4*Kr*K*Lr
               + 2*K*L*KK  + 2*Kr*Kl*L + 4*K*Kl*Lr +   K*K*LL
               - 8*K*Lr*KK - 4*K*Kl*LL + 4*KK*KK*L 
                 )/(n*n)

def KSE(X, Y, k):

    K3x = CSIC(X,X,k,k)
    K3y = CSIC(Y,Y,k,k)


    K = k(X,Y)
    n = len(K)
    Kr = K.sum(axis=1, keepdims=True) / n
    Kl = K.sum(axis=0, keepdims=True) / n
    KK = np.sum(K)/(n*n)
    
    K3xy = np.sum( K*K*K   + 3*K*K*KK  + 4*K*KK*KK - 3*K*K*Kl 
               - 3*K*K*Kr  + 6*K*Kl*Kr - 6*K*Kl*KK + 2*K*Kl*Kl
               - 6*K*Kr*KK + 2*K*Kr*Kr
                  )/(n*n)

    return K3x + K3y - 2*K3xy

def random_derangement(n):
    while True:
        v = np.arange(n)
        for j in np.arange(n - 1, -1, -1):
            p = np.random.randint(0, j+1)
            if v[p] == j:
                break
            else:
                v[j], v[p] = v[p], v[j]
        else:
            if v[0] != 0:
                return v

def independence_permutation_test(test_statistic, X, Y, num_permutations, prog_bar=True):
    statistics = np.zeros(num_permutations)
    range_ = range(num_permutations)
    if prog_bar:
        range_ = tqdm(range_)
    for i in range_:
        perm_inds = random_derangement(Y.shape[0])
        Y_ = Y[perm_inds]
        my_test_statistic = test_statistic(X, Y_)
        statistics[i] = my_test_statistic
    return statistics

def getResults(X, Y, mmd, n, alphas, perm_test = two_sample_permutation_test):
    if (X.ndim == 1):
        X = X[:,np.newaxis]
    if (Y.ndim == 1):
        Y = Y[:,np.newaxis]
        
    statistics = perm_test(mmd, X, Y, n, prog_bar=False)
    my_statistic = mmd(X,Y)
    ret = []
    for a in alphas:
        ret.append( my_statistic > np.percentile(statistics, 100 - a) )
    return ret

#### ---- synthetic data

def varUniform(a,b): 
    return (2.0/3.0)*(b*b + b*a + a*a)
    
def inverseVarUniform(a, V): 
    return -.5*a + np.sqrt(1.5 * V - 0.75 * a*a )

def generateUniforms(N, theta, a):
    b = inverseVarUniform(a, varUniform(0,1.0))
    X = np.random.uniform(-1.0, 1.0, N)
    Y = np.concatenate((np.random.uniform(-b, -a, round(theta*N)), np.random.uniform(a, b, N - round(theta*N))))
    return X,Y

def generateUniformAndGaussian(N, rho = 0.2):
    X = np.random.uniform(0.0, 1.0, (N, 1))
    Z = np.random.randn(N, 1)
    cutoff = round(rho*N)
    Y = np.vstack((norm.ppf(X[:cutoff]),Z[cutoff:]))
    return X,Y

def generateUniformAndChi2(N, rho = 0.2):
    X,Y = generateUniformAndGaussian(N, rho)
    return X, Y*Y

#### ---- testing

TWOSAMPLEKWARGS = {'perm_test' : two_sample_permutation_test, 
                   'mmd1'      : KME, 
                   'mmd2'      : KVE}

INDEPKWARGS     = {'perm_test' : independence_permutation_test,
                   'mmd1'      : lambda x,y,gk : HSIC(x,y,gk,gk), 
                   'mmd2'      : lambda x,y,gk : CSIC(x,y,gk,gk)}

def iteratedTesting(bWs, 
                    dataGenerator, 
                    N           = 25, 
                    num_tests   = 500, 
                    num_perms   = 500, 
                    perm_test   = two_sample_permutation_test,
                    mmd1        = KME, 
                    mmd2        = KVE, 
                    ker1        = gauss_kernel, 
                    ker2        = gauss_kernel, 
                    useTqdm     = True):
    res = []
    iterBands = tqdm(bWs) if useTqdm else bWs
    for bandwidth in iterBands:
        gK1 = lambda x,y : ker1(x,y, bandwidth)
        gK2 = lambda x,y : ker2(x,y, bandwidth)
        stat1 = lambda x,y : mmd1(x,y, gK1)
        stat2 = lambda x,y : mmd2(x,y, gK2)

        s1 = []
        s2 = []

        for i in range(num_tests):
            X, Y = dataGenerator(N)
            s1res = getResults(X, Y, stat1, num_perms, [5], perm_test = perm_test)
            s2res = getResults(X, Y, stat2, num_perms, [5], perm_test = perm_test)

            s1.append(s1res[0])
            s2.append(s2res[0])

        res.append((bandwidth, sum(s1)/num_tests, sum(s2)/num_tests))
    return res

def standardise(X, axis=0): 
    ran = X.max(axis = axis, keepdims = True) - X.min(axis = axis, keepdims = True)
    ran[ran==0.0] = 1.0
    return (X-X.min(axis = axis, keepdims = True)) / ran 
    
def cat2num(S):
    labs = S.unique()
    d = dict(zip(labs,range(len(labs))))
    return list(d[s] for s in S)

def dataGen(N, X, Y, std=False, toBox = True):
    indsX = np.random.randint(X.shape[0], size=N)
    indsY = np.random.randint(Y.shape[0], size=N)
    if std:
        return standardise(X[indsX], toBox = toBox), standardise(Y[indsY], toBox = toBox)
    return X[indsX], Y[indsY]

def getMeanSupOverBandwidths(input):
    arr = np.array(input)
    means = np.array([arr[:,:,:,1].mean(axis=1), arr[:,:,:,2].mean(axis=1)])
    means = means.transpose() 
    i = np.argmax(means, axis=0)
    res1, res2 = [], []
    for n, (amax1, amax2) in enumerate(i):
        res1.append(arr[n,:,amax1,1])
        res2.append(arr[n,:,amax2,2])
    return np.array(res1), np.array(res2)
   
def plotRes(dataKME, dataKVE, Nrange, ylabel = 'Test power (%)', maxLines = 110):
    def plotRange(data, lab, col):
        plt.plot(np.arange(1, len(data)+1, 1), 100.0 * data.mean(1), col+'-', alpha = 0.5)
        return plt.boxplot(100.0 * data.transpose(), 
                           medianprops  = dict(color = col, alpha = 0.5),
                           whiskerprops = dict(color = col),
                           capprops     = dict(color = col),
                           boxprops     = dict(color = col), 
                           flierprops   = dict(markeredgecolor = col, alpha = 0.75),
                           labels       = list(Nrange)) 
    
    fig = plt.figure()
    plotRange(dataKME, 'KME', 'r')
    plotRange(dataKVE, 'KVE', 'b')

    plt.xlabel('N', fontsize=16)
    plt.ylabel(ylabel, fontsize=16)   
    plt.hlines(list(range(0,maxLines,10)), linestyles='-', xmin=0.0, xmax=15.0, alpha=0.1)
    return fig

def sampleRangeExperiment(dataGen, 
                          Nrange, 
                          num_tests = 100, 
                          num_perms = 100, 
                          num_exps  = 5,
                          perm_test = two_sample_permutation_test, 
                          mmd1      = KME, 
                          mmd2      = KVE,
                          kernel    = gauss_kernel):
    res_tot = []
    for N in tqdm(Nrange):
        getResults = lambda : iteratedTesting(BANDWIDTHS, 
                                              dataGen, 
                                              N         = N, 
                                              num_tests = num_tests, 
                                              num_perms = num_perms, 
                                              perm_test = perm_test,
                                              mmd1      = mmd1, 
                                              mmd2      = mmd2,
                                              ker1      = kernel,
                                              ker2      = kernel,
                                              useTqdm   = False
                                              )
        res = []
        for _ in range(num_exps):
            res.append(getResults())
        res_tot.append(res)
    return getMeanSupOverBandwidths(res_tot)

def compareStatisticsCompTime(Nrange, stat1 = KME, stat2 = CSIC, num_aver = 10):
    import time
    def timer(f):
        timestamp = time.time()
        f()
        return time.time() - timestamp

    times_1 = []
    times_2 = []
    for N in Nrange:
        X = np.random.uniform(-1.0, 1.0, (N, 1))
        Y = np.random.uniform(-1.0, 1.0, (N, 1))
        time_1 =[]
        time_2 = []
        for _ in range(num_aver):
            time_1.append(timer(lambda : stat1(X,Y,gauss_kernel)))
            time_2.append(timer(lambda : stat2(X,Y,gauss_kernel)))
        times_1.append(np.mean(time_1))
        times_2.append(np.mean(time_2))
    fig = plt.figure()
    plt.plot(Nrange, times_1, 'r')
    plt.plot(Nrange, times_2, 'b')
    plt.xlabel('N', fontsize=16)
    plt.ylabel('Average times (s)', fontsize=16)   
    
    return fig