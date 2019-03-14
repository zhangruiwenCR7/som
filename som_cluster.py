#!/usr/bin/python
#--- coding:utf-8

import time
import sys
import matplotlib
matplotlib.use('Agg')
import numpy as np
from matplotlib import pyplot as plt

class SOM():
    def __init__(self, in_layer, s, out_size=(3,3), m_iter=1000):
        self.in_layer = in_layer.copy()
        self.m_iter = m_iter
        self.w = np.random.rand(out_size[0], out_size[1], self.in_layer.shape[1])
        self.color = ['y', 'r', 'g', 'b', 'c', 'm', 'k', 'pink', 'dark', 'orange', 'tan', 'gold']
        self.label = []
        self.res =[]
        self.neuron = {}
        self.l = 0 
        self.s = s
        
    def Normalize_Input(self, X):
        '''
        print X[1]
        for i in range(X.shape[0]):
            t = np.linalg.norm(X[i])
            X[i] /= t
            if i == 1:
                print t, X[1]
        #'''
        return X

    def Normalize_W(self, w):
        '''
        for i in range(w.shape[0]):
            for j in range(w.shape[1]):
                t = np.linalg.norm(w[i,j])
                w[i,j] /= t
        #'''
        return w
    
    def Get_Win_Neuron(self, x):
        max_dis=100000000
        for i in range(self.w.shape[0]):
            for j in range(self.w.shape[1]):
                #dis = x.dot(self.w[i,j]) #余弦距离
                dis = np.linalg.norm(x-self.w[i,j])#欧式距离
                if dis < max_dis:
                    max_dis = dis
                    win_index = (i,j)
        return win_index

    def Get_Neighborhood(self, win, radius):
        res = []
        for i in range(self.w.shape[0]):
            for j in range(self.w.shape[1]):
                dis = (np.linalg.norm([i-win[0], j-win[1]]))
                if dis <= radius:
                    res.append([i, j, dis])
        return res

    def Update_W(self, index, time, X, r):
        for i in range(len(index)):
            self.w[index[i][0],index[i][1]] += self.Learning_Rate(index[i][2], time, r)*(X-self.w[index[i][0],index[i][1]])

    def Radius(self, t):
        return int(self.w.shape[0]*(1-float(t)/self.m_iter)/2)+1

    def Learning_Rate(self, dis, time, r):
        #self.l = 0.6*np.exp(1/(time-self.m_iter))*np.exp(-float(dis)/r)
        self.l = 0.6*(1.0/(time+1))*np.exp(-float(dis)/r)
        return self.l

    def Get_Result(self):
        self.w = self.Normalize_W(self.w)
        self.neuron={}
        for i in range(self.in_layer.shape[0]):
            win = self.Get_Win_Neuron(self.in_layer[i])
            key = win[0]*self.w.shape[0] + win[1]
            self.label.append(key)
        
            if self.neuron.has_key(key):
                self.neuron[key].append(i)
            else:
                self.neuron.fromkeys([key])
                self.neuron[key]=[]
                self.neuron[key].append(i)
        print ' cluster result', self.neuron
        
    def Train(self, learning_rate=1, threshold=1):
        self.in_layer = self.Normalize_Input(self.in_layer)
        for i in range(self.m_iter):
            if i%(self.m_iter/10) == 0 and i >=0: print ' iteration:', i
            self.w = self.Normalize_W(self.w)
            for k in range(self.in_layer.shape[0]):
                j = np.random.randint(self.in_layer.shape[0])
                win = self.Get_Win_Neuron(self.in_layer[j])
                r = self.Radius(i)
                index = self.Get_Neighborhood(win, r)
                self.Update_W(index, i, self.in_layer[j], r)
                #print self.in_layer[j]
        self.Get_Result()
        return self.w.reshape(self.w.shape[0]*self.w.shape[1], self.w.shape[2])[self.neuron.keys()], self.neuron

    def Cluster(self):
        print 'a'
        

    def Draw_SOM_Grid(self, f0, f1, fname):
        '''
        for key in self.neuron.keys():
            i, j = key / self.w.shape[0], key % self.w.shape[0]
            plt.scatter(i, j, c=self.color[int(self.s[self.neuron[key][0]])])
            #plt.scatter(i, j, c='b')
        plt.savefig(fname+'-som-2D')
        plt.close()
        #'''
        for key in self.neuron.keys():
            i, j = key / self.w.shape[0], key % self.w.shape[0]
            f0.scatter(i, j, c=self.color[int(self.s[self.neuron[key][0]])], marker='.')
            f1.scatter(i, j, c='b', marker='.')
        f0.set_title('som-2D')
        f0.axis('off')
        f1.set_title('som-2D-Nolable')
        print ' Image saved OK!'

    def Draw_Normalize(self):
        plt.scatter(self.in_layer[:,0], self.in_layer[:,1], c=np.zeros(len(self.s)))
        plt.scatter(self.in_layer[:,0], self.in_layer[:,1], c=self.s)
        plt.savefig('t')
        plt.close()
        print ' Image saved OK!'

def main(argv):
    fname = argv[0]
    size = int(argv[1])
    iteration = int(argv[2])
    dataset = np.loadtxt('./dataset/'+fname+'.txt')
    print ' The shape of dataset:',dataset.shape
    if dataset.shape[1]==2: dataset = np.hstack((dataset, np.zeros((dataset.shape[0], 1))))
    
    image =plt.figure(figsize=(10, 9.5))
    ax = image.subplots(2,2)
    #f.figsize=(9,8)
    ax[0][0].scatter(dataset[:,0], dataset[:,1], c=dataset[:,2], marker='.')
    ax[0][0].set_title('source')
    ax[0][0].axis('off')
    ax[1][0].set_title('cluster')
    #ax[1][0].axis('off')
    '''
    plt.figure(figsize=(10, 9))
    plt.scatter(dataset[:,0], dataset[:,1], c=dataset[:,2])
    plt.title(fname+'dataset')
    plt.savefig(fname+'-dataset')
    plt.close()
    #'''

    som = SOM(dataset[:, :2], dataset[:, 2], (size, size), iteration)
    out, res = som.Train()
    som.Draw_SOM_Grid(ax[0][1], ax[1][1], fname)
    #som.Draw_Normalize()
    

    image.savefig(fname+'-dataset')
    plt.close()
    print len(res)

if __name__ == '__main__':
    print '\n ##--Start---- By Zhang Ruiwen ---', time.asctime( time.localtime(time.time()) ), '------\n' 
    main(sys.argv[1:])
    print '\n ##--End------ By Zhang Ruiwen ---', time.asctime( time.localtime(time.time()) ), '------\n' 