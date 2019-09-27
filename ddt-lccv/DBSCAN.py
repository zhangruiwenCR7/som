# -------------------------------------------------------------------------
# function: classes,type = dbscan(x, k, Eps)
# -------------------------------------------------------------------------
# Objective:
# Clustering the data with Density - Based Scan Algorithm with Noise (DBSCAN)
# -------------------------------------------------------------------------
# Input:
# x - dataset(m, n) m - objects, n - variables
# k - number of objects in a neighborhood of an object
# (minimal number of objects considered as a cluster)
# Eps - neighborhood radius, if not known avoid this parameter or put[]
# -------------------------------------------------------------------------
# Output:
# classes - vector specifying assignment of the i-th object to certain
# cluster(m, 1)
# type - vector specifying type of the i-th object
# (core: 1, border: 0, outlier: -1)
# -------------------------------------------------------------------------
# Example of use:
# x = [randn(30, 2)*.4 randn(40, 2)*.5 + ones(40, 1)*[4 4]]
# classes,type = dbscan(x, 5, [])
# -------------------------------------------------------------------------
# References:
# [1] M.Ester, H.Kriegel, J.Sander, X.Xu, A density - based algorithm for
# discovering clusters in large spatial databases with noise, proc.
# 2nd Int.Conf.on Knowledge Discovery and Data Mining, Portland, OR, 1996,
# p.226, available from:
# www.dbs.informatik.uni - muenchen.de / cgi - bin / papers?query = --CO
# [2] M.Daszykowski, B.Walczak, D.L.Massart, Looking for
# Natural Patterns in Data.Part 1: Density Based Approach,
# Chemom.Intell.Lab.Syst. 56(2001) 83 - 92
# -------------------------------------------------------------------------
# Written by Michal Daszykowski
# Department of Chemometrics, Institute of Chemistry,
# The University of Silesia
# December 2004
# http://www.chemometria.us.edu.pl

from numpy import min, abs, max, sqrt, arange, prod, pi, ones, insert, sum, zeros, argwhere, empty, append, delete
from scipy.special import gamma

# ...........................................
def epsilon(x, k):
    # function: epsi = epsilon(x, k)
    #
    # Objective:
    # Analytical way used for estimating neighborhood radius for the DBSCAN algorithm
    #
    # Input:
    # x - data matrix(m, n); m - data points, n - dimensions
    # k - number of data points in a neighborhood of a given data point
    #     (minimal number of data points considered as a cluster)
    m, n = x.shape
    maxmin = max(x, axis = 0) - min(x, axis = 0)
    epsi = ((prod(maxmin)*k*gamma(0.5*n + 1))/(m*sqrt(pi**n)))**(1./n)
    return epsi

# ............................................
def edist(i, x):
    # function: D = edist(i, x)
    #
    # Objective:
    # Calculate the Euclidean distances between the i-th sample vector and all m sample vectors in x
    #
    # Input:
    # i - an n-dimensional sample vector (1, n)
    # x - sample matrix (m, n); m - sample vector, n - dimension
    #
    # Output:
    # D - Euclidean distance(m, 1)
    m, n = x.shape
    if n == 1:
        D = abs(ones((m, 1))*i - x)
    else:
        squ = (ones((m, 1))*i - x)**2
        D = sqrt(sum(squ, axis = 1))
    return D

def dbscan(x, k, Eps):
    m, n = x.shape
    if len(Eps) != 1: Eps = epsilon(x, k)

    x = insert(x, 0, arange(m), 1)
    m, n = x.shape
    p_type = zeros((m, 1))
    classes = -ones((m, 1))
    no = 1
    touched = zeros((m, 1))

    for i in range(m):
        if touched[i] == 0:
            ob = x[i, :]
            D = edist(ob[1:n], x[:, 1:n])
            ind = argwhere(D <= Eps)

            if len(ind) > 1 and len(ind) < k + 1:     # do not deal with
                p_type[i] = 0
                classes[i] = 0

            if len(ind) == 1:          # this is noise
                p_type[i] = -1
                classes[i] = -1
                touched[i] = 1

            if len(ind) >= k + 1:      # make clustering
                p_type[i] = 1
                for j in range(len(ind)):
                    classes[ind[j]] = max(no)
                
                while len(ind) >= 1:
                    ob = x[int(ind[0]), :]
                    touched[int(ind[0])] = 1
                    ind = ind[1:len(ind)]
                    
                    
                    D = edist(ob[1:n], x[:, 1:n])
                    i1 = argwhere(D <= Eps)

                    if len(i1) > 1:
                        for j in range(len(i1)):
                            classes[i1[j]] = no
                        if len(i1) >= k + 1:
                            p_type[int(ob[0])] = 1
                        else:
                            p_type[int(ob[0])] = 0

                        for i in range(len(i1)):
                            if touched[i1[i]] == 0:
                                touched[i1[i]] = 1
                                ind = append(ind, i1[i])
                                
                                classes[i1[i]] = no
                no = no + 1
    no = no - 1
    i1 = argwhere(classes == 0)
    classes[i1] = -1
    cl = classes.transpose()
    p_type[i1] = -1
    return cl, p_type, no