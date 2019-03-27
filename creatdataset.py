#!/usr/bin/python3
#--- coding:utf-8

import time
import sys
import matplotlib
matplotlib.use('Agg')
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.datasets import samples_generator as sg


def Generator():
    #center=[[1,1,1],[3,3,3],[3,3,1]]
    #X,labels=sg.make_blobs(n_samples=500,centers=center,n_features=3,cluster_std=0.5,random_state=0)
    #X,labels=sg.make_blobs(n_samples=[80,90,100,110,120],centers=None,n_features=3,cluster_std=0.5,random_state=0)
    #X,labels=sg.make_classification(n_samples=400, n_features=3, n_informative=3, n_redundant=0, n_repeated=0, n_classes=4, n_clusters_per_class=1, weights=None, flip_y=0.01, class_sep=1.0, hypercube=True, shift=0.0, scale=1.0, shuffle=True, random_state=0)
    #X,labels=sg.make_gaussian_quantiles(mean=[0,4,6], cov=1.0, n_samples=300, n_features=3, n_classes=3, random_state=3)
    #X,labels=sg.make_multilabel_classification(n_samples=300, n_features=3, n_classes=5, n_labels=2)
    
    X,labels=sg.make_s_curve(n_samples=800, noise=0.0, random_state=3)
    Y=np.zeros_like(X)
    Y[:,0], Y[:,1], Y[:,2] = X[:,0]+0.5, X[:,1], X[:,2]+2.5
    X = np.hstack((X, np.zeros(len(X)).reshape(len(X),1)))
    Y = np.hstack((Y, (np.zeros(len(Y))+1).reshape(len(Y),1)))
    dataset = np.vstack((X,Y))
    '''
    X,labels=sg.make_swiss_roll(n_samples=500, noise=0.0, random_state=3)
    Y,labels=sg.make_swiss_roll(n_samples=500, noise=0.0, random_state=7)
    Y = -Y
    Y[:,1] = Y[:,1]+20
    X = np.hstack((X, np.zeros(len(X)).reshape(len(X),1)))
    Y = np.hstack((Y, (np.zeros(len(Y))+1).reshape(len(Y),1)))
    dataset = np.vstack((X,Y))
    print('X.shape', X.shape)
    '''
    #print("labels",set(labels))
    #dataset = np.hstack((X, labels.reshape(len(labels), 1)))
    return dataset

def Sphere():
    dataset=[]
    radius=[3,4,5]
    center=[5,5,5]
    for r in radius:
        X=np.random.rand(50)*r*2+5-r
        for a in X:
            Y=np.zeros(30)
            Y[00:15] = 5 + np.random.rand(15)*np.sqrt(r*r-(a-5)**2)
            Y[15:30] = 5 - np.random.rand(15)*np.sqrt(r*r-(a-5)**2)
            for b in Y:
                c = 5 - np.sqrt(r*r-(a-5)**2-(b-5)**2)
                dataset.append([a,b,c,r])
    return np.array(dataset)


def main(argv):
    fname = argv[0]

    #dataset=Generator()
    dataset=Sphere()
    print(dataset.shape)
    
    np.savetxt('./dataset/'+fname+'.txt', dataset, fmt='%.4f', delimiter=' ')

    fig =plt.figure(figsize=(10, 9.5))
    ax0 = fig.add_subplot(111, projection='3d')
    ax0.scatter(dataset[:,0], dataset[:,1], dataset[:,2], c=dataset[:,3])
    #ax0.view_init(elev=0,azim=0)
    ax0.view_init(elev=90,azim=0)
    ax0.set_zlim(0,10)
    plt.savefig(fname)
    plt.close()


if __name__ == '__main__':
    start_second = time.time()
    print('\n ##--Start-- By Ruiwen --', time.asctime(time.localtime(start_second)), '------#\n')
    main(sys.argv[1:])
    run_second = time.time() - start_second
    s = int(run_second % 60)
    m = int(run_second / 60)
    if m==0: rt='00:'+str(s)
    else: rt=str(m)+':'+str(s)
    print('\n ##--End---- By Ruiwen --', time.asctime(time.localtime(time.time())), '--', rt, 's--#\n')
