#!/usr/bin/python3
#--- coding:utf-8

import time, sys, os
import numpy as np
from IPython import embed
from sklearn.neighbors import KDTree
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

def main(argv):
    fname = argv[0]
    dataset = np.loadtxt('./dataset/'+fname+'.txt')
    tree = KDTree(dataset[:,:2], leaf_size=2)

    r=1
    flag=0
    NaN_Edge = np.zeros((dataset.shape[0], dataset.shape[0]))
    NaN_Num = np.zeros(dataset.shape[0])
    cnt_last = 0
    rep=-1
    
    while flag==0:
        for i in range(dataset.shape[0]):
            ind = tree.query([dataset[i,:2]], r, return_distance=False)[0][-1]
            temp = dataset[ind, :2]
            #embed(header='First time')
            ind_l = tree.query([temp], r, return_distance=False)[0]
            if i==0 or i==1: print(i, ind, ind_l)
            if (i in ind_l) and (NaN_Edge[i,ind]+NaN_Edge[ind,i]==0) and (i != ind):
                NaN_Edge[i,ind]=NaN_Edge[ind,i]=1
                NaN_Num[i] += 1
                NaN_Num[ind] += 1
        cnt = np.argwhere(NaN_Num==0).shape[0]
        if cnt == cnt_last: rep += 1
        else:               cnt_last = cnt
        if cnt == 0 or rep >= np.sqrt(r-rep): flag = 1
        r += 1
    e = r-1
    print(e)
    print(NaN_Num)
    #print(NaN_Edge)
    #print(len(np.argwhere(NaN_Edge>0)))
    color=['r','g','b','y']
    plt.scatter(dataset[:,0], dataset[:,1], c=dataset[:,2], marker='.')
    for i in range(dataset.shape[0]):
        for j in range(i+1, dataset.shape[0]):
            if not NaN_Edge[i, j]==0:
                plt.plot([dataset[i,0],dataset[j,0]], [dataset[i,1],dataset[j,1]], c='r')
                #plt.scatter(dataset[i,0], dataset[i,1], c=color[int(dataset[i,2])], marker='.')
    plt.savefig(fname+'-NaN')
    

if __name__ == '__main__':
    start_second = time.time()
    print('\n ##--Start-- By Ruiwen --', time.asctime(time.localtime(start_second)), '------#\n')
    main(sys.argv[1:])
    run_second = time.time() - start_second
    s = int(run_second % 60)
    m = int(run_second / 60)
    if m==0: rt='00:'+str(s)
    else:    rt=str(m)+':'+str(s)
    print('\n ##--End---- By Ruiwen --', time.asctime(time.localtime(time.time())), '--', rt, 's--#\n')