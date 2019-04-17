#!/usr/bin/python3
#--- coding:utf-8

import time, sys, os
import numpy as np
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
        W = np.zeros((len(X), len(X)))
        for i in range(len(X)):
            index = np.argpartition(S[i], self.knn)[:self.knn+1]
            temp = np.exp(-S[i,index]/self.sigma)
            W[i, index]=temp
        return W

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
        temp = np.linalg.norm(vec, axis=1)
        temp = np.repeat(np.transpose([temp]), self.n_clusters, axis=1)
        vec = vec / temp
        return Kmeans(self.n_clusters).fit(vec)
    
from sklearn.cluster import SpectralClustering as SC
def main(argv):
    fname = argv[0]
    d = int(argv[1])
    n = int(argv[2])
    dataset = np.loadtxt('./dataset/'+fname+'.txt')
    SC = SpectralClustering(n,10)
    labels = SC.fit(dataset[:,:d])
    '''
    #labels = SC(n).fit(dataset[:,:d]).labels_
    '''
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