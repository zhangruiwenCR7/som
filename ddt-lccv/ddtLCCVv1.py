## @Copyright by DENG, Zhidong, Department of Computer Science, Tsinghua University
## Updated on April 6, 2019
## Density-based Distance Tree (DDT) + LCCV

import sys
import numpy as np
import pylab as pl
from matplotlib import cm
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import networkx as nx
from sklearn.cluster import SpectralClustering
from sklearn.cluster import DBSCAN
import DBSCAN as ds
import LCCV

"""
def ELC(data, nclust):
    #:param X: 2D matrix of N*D, i.e. N input sample data with D dimension
    #:param nclust: specific number of clusters
    #:return: the best clustering label
    #gamma = 0.2           # convex clusters
    gamma = 10             # non-convex cluster
    bestcl = SpectralClustering(n_clusters=nclust, gamma=gamma).fit_predict(data)
    return bestcl
"""

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def ELC(A):
    """
    :param A: 2D matrix of N*D, i.e. N input sample data with D dimension
    :param nclust: specific number of clusters
    :return: the best clustering label
    """
    N, dim = A.shape
    # Compute the Euclidean distance matrix between pairwise points
    dist = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            for k in range(dim):
                dist[i, j] +=  (A[i, k] - A[j, k])**2
            dist[i, j] = np.sqrt(dist[i, j])
    # Sort the elements of each row vector of distance matrix in ascending order
    sdist = np.sort(dist, axis = 1)
    index = np.argsort(dist, axis = 1)

    #################################################################################################
    # The Natural Neighbor Searching algorithm
    print ('Start running the Natural Neighbor Searching algorithm ...')
    r = 1
    flag = 0
    nb = np.zeros((N, 1))    #The number of reverse neighbor of each data point
    count = 0
    count1 = 0
    while (flag == 0):
        for i in range(N):
            k = index[i, r]
            nb[k] += 1
            #RNN(k, nb[k]) = i
        r += 1
        count2 = 0
        for i in range(N):
            if nb[i] == 0:
                count2 += 1
        if count1 == count2:   # If it remains unchanged
            count += 1
        else:
            count = 1
        if (count2 == 0) or ((r > 2) and (count >= 2)):           #The terminal condition
            flag = 1
        count1 = count2

    supk = r - 1                                                  #The characteristic value of natural neighbor
    max_nb = int(np.max(nb))                                      #The maximum value of nb
    print ('The characteristic value is %d' % supk)
    print ('The maximum value of nb is %d' % max_nb)

    #################################################################################################
    # Calculate the local density of each data point
    rho = np.zeros((N, 1))
    for i in range(N):
         d = 0
         for j in range(max_nb + 1):
             d += sdist[i, j]
         rho[i] = max_nb / d
    #################################################################################################
    # The LORE Algorithm
    rho_sorted = np.array(sorted(rho, reverse = True))      #sort local density in a descending order
    reverse_index = np.lexsort((rho_sorted, rho), axis = 0)
    ordrho = reverse_index[::-1]
    print ('Start running the LORE algorithm ...')
    local_core = -np.ones((N, 1))                   #the local core or representatives of each data point
    for i in range(N):
        p = ordrho[i]
        maxrho = rho[p]
        maxindex = p
        # Find the data point with maximum density in local neighbors
        for j in range(int(nb[p]) + 1):
            x = index[p, j]
            if maxrho < rho[x]:
                maxrho = rho[x]
                maxindex = x
        # Assign representatives of the data points using maximum density
        if local_core[maxindex] == -1:
            local_core[maxindex] = maxindex
        # Assign representatives in local neighbors
        for j in range(int(nb[p]) + 1):
            if local_core[index[p, j]] == -1:
                local_core[index[p, j]] = local_core[maxindex]
            # Determine the representatives according to RCR
            else:
                q = local_core[index[p, j]]
                if dist[index[p, j], int(q)] > dist[index[p, j], int(local_core[maxindex])]:    #rho(q) < rho(local_core(maxindex))
                    local_core[index[p, j]] = local_core[maxindex]
            # Determine the representatives according to RTR
            for m in range(N):
                if local_core[m] == index[p, j]:
                    local_core[m] = local_core[index[p, j]]
    #print(local_core)
    # Find the cores or initial clusters
    cluster_number = 0
    cores0 = -np.ones((N, 1))
    cl = np.zeros((N, 1))
    for i in range(N):
        if local_core[i] == i:
            cores0[cluster_number] = i
            cluster_number += 1
            cl[i] = cluster_number
    cores = cores0[0:cluster_number]
    for i in range(N):
        cl[i] = cl[int(local_core[i])]
    
    print ('The number of initial clusters is %d' % cluster_number)

    #################################################################################################
    # Draw the local cores and the the initial clustering results
    # plot(A(:, 1), A(:, 2), '.')
    # hold on
    # for i=1:N
    # plot([A(i, 1), A(local_core(i), 1)], [A(i, 2), A(local_core(i), 2)])
    # hold on
    # end
    # drawcluster2(A, cl, cluster_number + 1)
    # hold on
    # plot(A(local_core, 1), A(local_core, 2), 'r*', 'MarkerSize', 8)

    #################################################################################################
    # Construct the adjacency matrix of graph
    weight = np.zeros((N, N))
    for i in range(N):
        for j in range(1, supk + 1):
             x = index[i, j]
             weight[i, x] = dist[i, x]

    ############################################################################################################
    # Compute the shortest path length or graph-based distance between the cores based on the Dijkstra algorithm
    print ('Start with computing the graph-based distance between cores ...')
    shortest_path = np.zeros((cluster_number, cluster_number))        #The shortest path for core-based graph
    #maxd = (np.exp(alpha * maxd) - 1)**(1 / alpha)
    #weight2 = sparse(weight)
    G = nx.DiGraph()
    for i in range(N):
        G.add_node(i)
        for j in range(N):
            if weight[i, j] == 0:  weight[i, j] = float('inf')
            G.add_weighted_edges_from([(i, j, weight[i, j])])
    for i in range(cluster_number):
        shortest_path[i, i] = 0
        Dict = nx.single_source_dijkstra_path_length(G, int(cores[i]))
        D = list(Dict.values())
        D_index = list(Dict.keys())
        #D = np.array(D).transpose()
        for j in range(i+1, cluster_number):
            indx_ = D_index.index(int(cores[j]))
            shortest_path[i, j] = D[indx_]
            if shortest_path[i, j] == float('inf'):
                shortest_path[i, j] = 0
            shortest_path[j, i] = shortest_path[i, j]
    maxd = np.max(np.max(shortest_path))
    for i in range(cluster_number):
        for j in range(cluster_number):
            if shortest_path[i, j] == 0:
                shortest_path[i, j] = maxd
    print ('Start with clustering ...')
    total_iter = int(np.floor(np.sqrt(N)))+1
    #lccv_indx = np.zeros((cluster_number, 1))
    lccv_indx = np.zeros((total_iter, 1))
    maxlccv_indx = -1
    for nc in range(2, total_iter):                              # 50%cluster_number - 1
        #--------------------------------------------------------------------------
        # Use the K-means algorithm to make clustering
        #cl = k_means(A, 'random', nc);
        #--------------------------------------------------------------------------
        # Use the DBSCAN algorithm to make clustering
        cl, p_type, no = ds.dbscan(A, nc, [])
        cl = cl.transpose()
        
        #[cl, ~, ~] = DBSCAN(A, nc)
        #cl = DBSCAN(eps=0.2, min_samples=nc).fit_predict(A)
        # --------------------------------------------------------------------------
        # Use the spectral density algorithm to make clustering
        #gamma = 0.2           # convex clusters
        #gamma = 10            # non-convex cluster
        #cl = SpectralClustering(n_clusters=nclust, gamma=gamma).fit_predict(data)
        # --------------------------------------------------------------------------
        for i in range(N):
            if (cl[i] != -1) and (cl[i] != cl[int(local_core[i])]):
                cl[i] = cl[int(local_core[i])]
        for i in range(int(np.max(cl))):
            temp1 = np.argwhere(cl == (i + 1))
            #if isempty(temp1):
            if len(temp1) == 0:
                for j in range(i, int(np.max(cl))):
                    temp2 = np.argwhere(cl == (j + 1))
                    #if ~isempty(temp2):
                    if len(temp2) != 0:
                        for k in range(len(temp2)):
                            cl[temp2[k]] = cl[temp2[k]] - 1
        ncl = int(np.max(cl))   # the number of clusters
        # --------------------------------------------------------------------------
        # Compute the local core clustering validity
        lccv_indx[nc] = LCCV.computeLCCV(A, cl, ncl, cores, shortest_path, local_core)
        print ('nc = %d   ncl = %d   LCCV = %f' % (nc, ncl, lccv_indx[nc]))
        # --------------------------------------------------------------------------
        if maxlccv_indx < lccv_indx[nc]:
            maxlccv_indx = lccv_indx[nc]
            bestcl = cl
    print ('Process ends')

    return bestcl

########################################################################################################
#                                               MAIN                                                   #
########################################################################################################
# d:/Python/dzd_1/dataset1/A.txt, B.txt, C.txt, D.txt, E.txt, F.txt
# d:/Python/dzd_1/dataset2/two_cluster.txt, Three_cluster.txt, ..., ThreeCircles.txt
# d:/Python/dzd_1/dataset3/sn.txt, ..., spiral.txt, ..., db3.txt, ..., E6.txt
# dataset3: sn, sk, fc1, line, spiral, circle, jain, db, db3, ls, cth, E6
if __name__ == '__main__':
    fname=sys.argv[1]
    data = np.loadtxt('../../04dataset/'+fname+'.txt')
    ND = data.shape[0]                            #number of sample data
    dim = data.shape[1]                           #dimension of input sample vector
    if dim == 3:                                  #2D sample vector + label
        data = np.delete(data, 2, axis = 1)     #have removal of label column
        dim = dim - 1
        #nclust = 3                                   #given number of clusters

    ##########################################################################
    # Use the LCCV-based hierarchical clustering algorithm 
    #bestcl = HC_LCCV(data)

    # Use the LCCV-based DBSCAN clustering algorithm
    bestcl = ELC(data)

    # Use the LCCV-based DBSCAN clustering algorithm (by dzd)
    #bestcl = DDT(data)
    # cl, p_type, no = ds.dbscan(data, 4, [])
    # bestcl = cl.transpose()
    classes = bestcl.copy()
    nclust = int(np.max(bestcl))
    print(nclust)

    ##########################################################################
    # Draw input sample data that are labeled with unsupervised learning 
    # pl.clf()
    # pl.setp(pl.gca(), xticks = [])
    # pl.setp(pl.gca(), yticks = [])

    start = 0.2
    stop = 0.8
    cm_subsection = np.linspace(start, stop, nclust)
    colors = [cm.jet(x) for x in cm_subsection]
    # use colors as indicators of different categories
    for i in range(ND):
        if classes[i] > 0:
            ic = int(classes[i]) - 1
            plt.plot(data[i, 0], data[i, 1], marker='.', markersize=15, markerfacecolor=colors[ic], markeredgecolor=colors[ic])
        else:
            plt.plot(data[i, 0], data[i, 1], marker='.', markersize=15, markerfacecolor='k', markeredgecolor='k')
    plt.savefig('./res2/'+fname)
    """        
    # draw every tree-like cluster
    for k in range(nclust):
        for i in range(nclust, ND):
            if (classes[parent[ordrho[i]]] == classes[ordrho[i]]) and (classes[ordrho[i]] == (k + 1)):
            xx = [data[parent[ordrho[i]], 0], data[ordrho[i], 0]]
            yy = [data[parent[ordrho[i]], 1], data[ordrho[i], 1]]
            pl.plot(xx, yy, linestyle='-', linewidth=2, color="#999999")
    # draw a relational tree among cluster heads
    for k in range(nclust - 1):
        npeak0 = int(peak_root[k, DIM])
        npeak1 = int(peak_root[k + 1, DIM])
        xx = [data[npeak1, 0], data[npeak0, 0]]
        yy = [data[npeak1, 1], data[npeak0, 1]]
        pl.plot(xx, yy, linestyle='-', linewidth=2.5, color="#000000")
    # the density peak as root node of a tree-like cluster
    pl.plot(peak_root[0, 0], peak_root[0, 1], marker='o', markersize=12, markerfacecolor="#FF0000", markeredgecolor='k')
    for i in range(1, nclust):
        pl.plot(peak_root[i, 0], peak_root[i, 1], marker='o', markersize=12, markerfacecolor="#BBBBBB", markeredgecolor='r')
        #pl.plot(data[ordrho[i], 0], data[ordrho[i], 1], marker='^', markersize=12, markerfacecolor="#999999", markeredgecolor='k')
        #pl.plot(data[632, 0], data[632, 1], marker='x', markersize=30, markerfacecolor="#00FFFF", markeredgecolor='k')
        #pl.plot(data[160, 0], data[160, 1], marker='x', markersize=30, markerfacecolor="#0000FF", markeredgecolor='k')
    """

    #pl.show()
