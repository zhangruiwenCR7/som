#!/usr/bin/python3
#--- coding:utf-8

import time, sys, os
from IPython import embed
import numpy as np
from sklearn.neighbors import KDTree
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

class Kmeans():
    def __init__(self, n_clusters, max_iter = 600, tol = 0.0001):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol

    def fit(self, data):
        shape,_ = data.shape
        index = np.random.randint(0,shape,size=self.n_clusters)
        k_points = data[index]
        k_points_last = None
        for a in range(self.max_iter):
            label = []
            k_points_last = k_points.copy()
            for i in range(shape):
                dis = []
                for j in range(self.n_clusters):
                    dis.append(np.linalg.norm(data[i,:]-k_points[j,:]))
                label.append(dis.index(min(dis)))
            for i in range(self.n_clusters):
                index = np.argwhere(np.array(label)==i)
                if len(index) != 0: k_points[i,:] = data[index, :].mean(axis=0)
            if np.linalg.norm(k_points-k_points_last) < self.tol:
                break
        return np.array(label)

class SpectralClustering():
    def __init__(self, n_clusters, knn=5, sigma=1.):
        self.n_clusters = n_clusters
        self.knn = knn
        self.sigma = sigma**2

    def Get_Dis(self, X):
        S = np.zeros((len(X), len(X)))
        for i in range(len(X)):
            for j in range(i+1, len(X)):
                S[i][j] = np.sum((X[i]-X[j])**2)
                S[j][i] =S[i][j]
        return S

    def Get_W_KNN(self, X, S):
        NaN_Edge = np.zeros((len(X), len(X)))
        #'''
        tree = KDTree(X, leaf_size=2)
        r=1
        flag=0
        NaN_Num = np.zeros(len(X))
        cnt_last = 0
        rep=-1
        
        while flag==0:
            for i in range(len(X)):
                ind = tree.query([X[i]], r, return_distance=False)[0][-1]
                temp = X[ind]
                ind_l = tree.query([temp], r, return_distance=False)[0]
                if (i in ind_l) and (NaN_Edge[i,ind]+NaN_Edge[ind,i]==0) and (i != ind):
                    temp = np.exp(-S[i,ind]/self.sigma)
                    NaN_Edge[i,ind]=NaN_Edge[ind,i]=temp
                    NaN_Num[i] += 1
                    NaN_Num[ind] += 1
            cnt = np.argwhere(NaN_Num==0).shape[0]
            if cnt == cnt_last: rep += 1
            else:               cnt_last = cnt
            if cnt == 0 or rep >= np.sqrt(r-rep): flag = 1
            r += 1
        '''
        for i in range(len(X)):
            index = np.argpartition(S[i], self.knn)[:self.knn+1]
            #print(index)
            temp = np.exp(-S[i,index]/self.sigma)
            NaN_Edge[i, index]=temp
        #'''
        return NaN_Edge

    def Get_D_L(self, W):
        D = np.diag(W.sum(axis=1))
        L = D - W
        d = np.linalg.inv(np.sqrt(D))
        l = np.dot(np.dot(d,L),d)
        return l

    def fit(self, X):
        S = self.Get_Dis(X)
        W = self.Get_W_KNN(X, S)
        L = self.Get_D_L(W)
        w, v = np.linalg.eig(L)
        index = np.argpartition(w, self.n_clusters)[:self.n_clusters]
        vec = v.real[:, index]
        '''
        temp = np.linalg.norm(vec, axis=1)
        temp = np.repeat(np.transpose([temp]), self.n_clusters, axis=1)
        vec = vec / temp
        '''
        print(vec)
        #embed(header='First time')
        from sklearn.cluster import KMeans
        return KMeans(self.n_clusters).fit(vec).labels_+1
        return Kmeans(self.n_clusters).fit(vec)
    
from sklearn.cluster import SpectralClustering as SC
def main(argv):
    fname = argv[0]
    d = int(argv[1])
    n = int(argv[2])
    dataset = np.loadtxt('./dataset/'+fname+'.txt')
    #'''
    SC = SpectralClustering(n, 4)
    labels = SC.fit(dataset[:,:d])
    '''
    labels = SC(n, n_neighbors=2).fit(dataset[:,:d]).labels_
    #'''
    plt.scatter(dataset[:,0], dataset[:,1], c=labels)
    plt.savefig(fname+'-spectral')


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