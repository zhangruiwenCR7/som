## @Copyright by DENG, Zhidong, Department of Computer Science, Tsinghua University
## Updated on April 6, 2019
## Density-based Distance Tree (DDT) + LCCV

import numpy as np

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Compute the Silhouette width criterion (SWC)
def computeSWC(D, cl_cores, ncl, shortpath):
    """
    :param D: 2D dataset of n*d, i.e. n the number of cores (initial clusters) with d dimension
    :param cl_cores: cluster label of each cores
    :param ncl: the number of clusters
    :param shortpath: the shortest path length matrix of core-based graph
    :return: swc: result the Silhouette width criterion (when using the Silhouette index)
             sl: the Silhouette value of each point (used for computing the LCCV index)
    """
    # -------------------------------------------------------------------------
    # Reference: R.P.Silhouettes: A graphical aid to the interpretation and
    # validation of cluster analysis[J].Journal of Computational & Applied
    # Mathematics, 1987, 20(20): 53-65.
    # -------------------------------------------------------------------------

    n, d = D.shape
    ncl = int(ncl)
    #cdata = cell(1, ncl)
    cdata = -np.ones((ncl, n*d+1))                    # keep data points in each cluster
    cindex = np.zeros((ncl, n))
    numc = ncl
    for i in range(ncl):
        nump = 0
        for j in range(n):
            if cl_cores[j] == (i+1):
                for k in range(d):
                    cdata[i, nump*d+k] = D[j, k]
                cindex[i, nump] = j
                nump += 1
        cdata[i, n*d] = nump
    numo = 0
    # Do not compute the swc for outliers
    if np.min(cl_cores) <= 0:
        for i in range(n):
            if cl_cores[i] <= 0:
                numo += 1
    swc = 0
    s1 = np.zeros((n, 1))
    for i in range(numc):
        aa = 0
        bb = 0
        ss = 0
        np1 = int(cdata[i, n*d])
        if np1 > 1:
            for j in range(np1):
                # compute aa
                suma = 0
                for k in range(np1):
                    if j != k:
                        suma += shortpath[int(cindex[i, j]), int(cindex[i, k])]
                aa = suma / (np1 - 1)
                # compute bb
                dd = np.ones((numc, 1)) * float('inf')
                for k in range(numc):
                    if k != i:
                        np2 = int(cdata[k, n*d])
                        #print(np2)
                        sumd = 0
                        for l in range(np2):
                            sumd += shortpath[int(cindex[i, j]), int(cindex[k, l])]
                        if np2 != 0: dd[k] = sumd / np2
                bb = np.min(dd, axis = 0)
                #print('bb:', bb)
                # compute ss
                if (np.max([aa, bb]) != 0):
                    if bb != float('inf'):
                        ss = (bb - aa) / np.max([aa, bb])
                    else:
                        ss = 0
                else:
                    ss = 0
                #print('np1=%d,numc=%d' % (np1, numc))
                #print('a(j)=%f,b(j)=%f,max(a,b)=%f,s(j)=%f' % (aa, bb, np.max([aa, bb]), ss))
                s1[int(cindex[i, j])] = ss
                swc = swc + ss
    #print ('swc=%f' % swc)
    if (n - numo) != 0:
        swc = swc / (n - numo)
    else:
        swc = 0

    return swc, s1

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Compute the LCCV index
def computeLCCV(A, cl, ncl, cores, shortpath, local_core):
    """
    :param A: 2D matrix dataset of N*D, i.e. N input sample data with D dimension
    :param cl：cluster label of each point
    :param ncl：the number of clusters
    :param cores：the initial clustering (part of data samples)
    :param shortpath：the shortest path length matrix between cores
    :param local_core: the representative of each data point
    :return: lccv: the clustering validity index
    Note: It is required to call the function computeSWC()
    """
    # -------------------------------------------------------------------------
    # Reference: R.P.xxx:
    # xxxxxx[J].IEEE Trans Neural Network and Learning System,
    # 2019, xx(xx): xx-xx.
    # -------------------------------------------------------------------------

    n, dim = A.shape
    ncores = cores.shape[0]
    D = np.zeros((ncores, dim))
    cl_cores = np.zeros((ncores, 1))
    for k in range(ncores):
        D[k, :] = A[int(cores[k]), :]
        cl_cores[k] = cl[int(cores[k])]
    # Count the number of points belonging to each cores
    nl = np.zeros((ncores, 1))
    for k in range(ncores):
        for j in range(n):
            if local_core[j] == cores[k]:
                if cl[j] >= 0:
                    nl[k] += 1
    # print(cores)
    # print(local_core)
    swc, s = computeSWC(D, cl_cores, ncl, shortpath)
    lccv = 0
    for k in range(ncores):
        lccv += s[k] * nl[k]
    lccv = lccv / n

    return lccv
