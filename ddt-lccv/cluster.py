#!/usr/bin/python3
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from matplotlib import cm
import DBSCAN as ds
from sklearn.cluster import DBSCAN

if __name__ == '__main__':
    fname=sys.argv[1]
    nc=int(sys.argv[2])
    # eps = float(sys.argv[3])
    data = np.loadtxt('/mnt/c/document/01research/01cluster/04dataset/'+fname+'.txt')
    ND = data.shape[0]
    dim = data.shape[1]                           #dimension of input sample vector
    if dim == 3:                                  #2D sample vector + label
        data = np.delete(data, 2, axis = 1)     #have removal of label column
        dim = dim - 1
    cl, p_type, no = ds.dbscan(data, nc, [])
    cl = cl.transpose()
    # cluster = DBSCAN(eps=eps, min_samples=nc).fit(data)
    # cl=cluster.labels_+1
    classes = cl.copy()
    nclust = int(np.max(cl))
    print(nclust)
    start = 0.2
    stop = 0.8
    cm_subsection = np.linspace(start, stop, nclust)
    colors = [cm.jet(x) for x in cm_subsection]
    for i in range(ND):
        if classes[i] > 0:
            ic = int(classes[i]) - 1
            plt.plot(data[i, 0], data[i, 1], marker='.', markerfacecolor=colors[ic], markeredgecolor=colors[ic])
        else:
            plt.plot(data[i, 0], data[i, 1], marker='.', markerfacecolor='k', markeredgecolor='k')
    plt.savefig('./dbscan/'+fname+'-'+sys.argv[2])
