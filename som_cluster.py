#!/usr/bin/python3
#--- coding:utf-8

import time
import sys, os
import matplotlib
matplotlib.use('Agg')
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

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
        radius=int(self.w.shape[0]/3)
        r = radius**2
        #radius = int(radius)
        #radius=30
        for i in range(max(0, win[0]-radius), min(self.w.shape[0], win[0]+radius+1)):
            for j in range(max(0, win[1]-radius), min(self.w.shape[1], win[1]+radius+1)):
                #dis = np.linalg.norm([i-win[0], j-win[1]])
                dis = (i-win[0])**2 + (j-win[1])**2
                if dis <= r:
                    res.append([i, j, dis])
        return res

    def Update_W(self, index, time, X, r):
        for i in range(len(index)):
            self.w[index[i][0],index[i][1]] += self.alpha(time)*self.h(index[i][2],r)*(X-self.w[index[i][0],index[i][1]])
            #self.w[index[i][0],index[i][1]] += self.Learning_Rate(index[i][2], time, r)*(X-self.w[index[i][0],index[i][1]])

    def Radius(self, t):
        #return min(int(self.w.shape[0]*(1-float(t)/self.m_iter)/2)+1, 20)
        #self.r = int(self.w.shape[0]*(0.8/(2*t+3)))+1
        self.r = self.w.shape[0]*(1.5/(t+3))
        return self.r

    def Learning_Rate(self, dis, time, r):
        #self.l = 0.6*np.exp(1/(time-self.m_iter))*np.exp(-float(dis)/r)
        self.l = 1.*(1.0/(4*time+1))*np.exp(-float(dis**2)/(r**2)/2)
        if self.l <= 0.01: self.l=0.01
        return self.l
    
    def alpha(self, t):
        self.l = 1./(2*t+1)
        if self.l <= 0.01: self.l = 0.01
        return self.l

    def h(self, dis, r):
        return np.exp(-float(dis)/(2*r*r))

    def Get_Result(self):
        self.w = self.Normalize_W(self.w)
        self.neuron={}
        for i in range(self.in_layer.shape[0]):
            win = self.Get_Win_Neuron(self.in_layer[i])
            key = win[0]*self.w.shape[0] + win[1]
            self.label.append(key)
            if key in self.neuron.keys():
                self.neuron[key].append(i)
            else:
                self.neuron.fromkeys([key])
                self.neuron[key]=[]
                self.neuron[key].append(i)
        #print(' cluster result:', self.neuron)
        
    def Train(self, fpath):
        self.in_layer = self.Normalize_Input(self.in_layer)
        for i in range(self.m_iter):
            if i%(self.m_iter/20) == 0 and i >=0: 
                print(' iteration:', i*100/self.m_iter, '%')
                self.Get_Result()
                img =plt.figure(figsize=(20, 10))
                ax2 = img.add_subplot(122)
                if self.w.shape[2]==2: ax1 = img.add_subplot(121)
                else: ax1 = img.add_subplot(121, projection='3d')
                for key in self.neuron.keys():
                    a, b = int(key / self.w.shape[0]), key % self.w.shape[0]
                    ax2.scatter(a, b, c=self.color[int(self.s[self.neuron[key][0]])], marker='.')
                    if self.w.shape[2] ==2:
                        ax1.scatter(self.w[a][b][0], self.w[a][b][1], c=self.color[int(self.s[self.neuron[key][0]])], marker='.')
                    else:
                        ax1.scatter(self.w[a][b][0], self.w[a][b][1], self.w[a][b][2], c=self.color[int(self.s[self.neuron[key][0]])], marker='.')
                img.savefig('./neuron-som-'+fpath+'/pro-'+str(i))
            self.w = self.Normalize_W(self.w)
            for k in range(self.in_layer.shape[0]):
                j = np.random.randint(self.in_layer.shape[0])
                win = self.Get_Win_Neuron(self.in_layer[j])
                r = self.Radius(i)
                index = self.Get_Neighborhood(win, r)
                self.Update_W(index, i, self.in_layer[j], r)
        self.Get_Result()
        return self.w.reshape(self.w.shape[0]*self.w.shape[1], self.w.shape[2])[list(self.neuron.keys())], self.neuron

    def Cluster(self):
        print('a')

    def Draw_SOM_Grid(self, f0, f1, fname):
        for key in self.neuron.keys():
            i, j = key / self.w.shape[0], key % self.w.shape[0]
            f0.scatter(i, j, c=self.color[int(self.s[self.neuron[key][0]])], marker='.')
            f1.scatter(i, j, c='b', marker='.')
        f0.set_title('som-2D-'+fname)
        f0.axis('off')
        f1.set_title('som-2D-'+str(self.w.shape[0]))

    def Draw_Normalize(self):
        plt.scatter(self.in_layer[:,0], self.in_layer[:,1], c=np.zeros(len(self.s)))
        plt.scatter(self.in_layer[:,0], self.in_layer[:,1], c=self.s)
        plt.savefig('t')
        plt.close()
        print(' Image saved OK!')

def main(argv):
    fname = argv[0]
    dataset = np.loadtxt('./dataset/'+fname+'.txt')

    if len(argv) > 1: dimension = int(argv[1])
    else:             dimension = 2

    if len(argv) > 2: size = int(argv[2])
    else: 
        #size = max(min(100, int(dataset.shape[0]/50)*5), 10)
        size = max(min(100, int(np.sqrt(dataset.shape[0])*2)), 10)

    if len(argv) > 3: iteration = int(argv[3])
    else:             iteration = 100

    if not os.path.exists('./neuron-som-'+fname):
        os.makedirs('./neuron-som-'+fname)

    print(' The shape of dataset:', dataset.shape)
    print(' The size of grid: ', size)
    if dataset.shape[1]==dimension: dataset = np.hstack((dataset, np.zeros((dataset.shape[0], 1))))
    
    v=0
    for i in range(dataset.shape[1]-1):
        edge = max(dataset[:,i]) - min(dataset[:, i])
        v += edge*edge
        print(edge)
    
    som = SOM(dataset[:, :dimension], dataset[:, -1], (size, size), iteration)
    
    fig =plt.figure(figsize=(10, 9.5))
    
    ax12 = fig.add_subplot(222)
    
    ax22 = fig.add_subplot(224)
    if dimension==2:
        ax11 = fig.add_subplot(221)
        for i in range(dataset.shape[0]):
            ax11.scatter(dataset[i,0], dataset[i,1], c=som.color[int(dataset[i,2])], marker='.')
            ax11.axis('off')
    elif dimension==3:
        ax11= fig.add_subplot(221, projection='3d')
        #ax11.view_init(elev=5,azim=75)
        for i in range(dataset.shape[0]):
            ax11.scatter(dataset[i,0], dataset[i,1], dataset[i,2], c=som.color[int(dataset[i,3])], marker='.')
    ax11.set_title('source-'+str(dataset.shape[0]))
    #ax11.axis('off')
    
    #ax[1][0].axis('off')

    out, res = som.Train(fname)
    som.Draw_SOM_Grid(ax12, ax22, str(len(res)))
    if dimension == 2:
        ax21 = fig.add_subplot(223)
        for key in som.neuron.keys():
            i, j = int(key / som.w.shape[0]), key % som.w.shape[0]
            ax21.scatter(som.w[i][j][0], som.w[i][j][1], c=som.color[int(som.s[som.neuron[key][0]])], marker='.')
    elif dimension == 3:
        ax21 = fig.add_subplot(223, projection='3d')
        for key in som.neuron.keys():
            i, j = int(key / som.w.shape[0]), key % som.w.shape[0]
            ax21.scatter(som.w[i][j][0], som.w[i][j][1], som.w[i][j][2], c=som.color[int(som.s[som.neuron[key][0]])], marker='.')
    ax21.set_title('neuron-'+str(len(res)))
    #som.Draw_Normalize()

    fig.savefig(fname+'-dataset-'+str(size))
    plt.close()
    print(len(res))

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