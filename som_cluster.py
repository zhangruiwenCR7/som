#!/usr/bin/python3
#--- coding:utf-8

import time, sys, os
import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class SOM():
    def __init__(self, in_layer, s, out_size=(3,3), m_iter=1000):
        self.in_layer = in_layer.copy()
        self.m_iter = m_iter
        a,b = np.min(self.in_layer), np.max(self.in_layer)
        self.w = (b-a)*np.random.rand(out_size[0], out_size[1], self.in_layer.shape[1])+a
        self.color = ['y', 'r', 'g', 'b', 'c', 'm', 'k', 'pink', 'dark', 'orange', 'tan', 'gold']
        self.label = []
        self.res =[]
        self.neuron = {}
        self.l = 1.0 
        self.s = s
        self.som_r = int(self.w.shape[0]/3)
        self.som_r_square = self.som_r**2
        self.D_list = []
    
    def Init_W(self):
        for i in range(self.w.shape[0]):
            for j in range(self.w.shape[1]):
                k = np.random.randint(self.in_layer.shape[0])
                self.w[i][j]=self.in_layer[k]
        
    def Normalize_Input(self, X):
        #'''
        for i in range(X.shape[0]):
            t = np.linalg.norm(X[i])
            X[i] /= t
        #'''
        return X

    def Normalize_W(self, w):
        #'''
        for i in range(w.shape[0]):
            for j in range(w.shape[1]):
                t = np.linalg.norm(w[i,j])
                w[i,j] /= t
        #'''
        return w
    
    def Get_Win_Neuron(self, x):
        max_dis=float('inf')
        min_dis=-float('inf')
        for i in range(self.w.shape[0]):
            for j in range(self.w.shape[1]):
                #'''
                dis = x.dot(self.w[i,j]) #余弦距离
                if dis > min_dis:
                    min_dis = dis
                    win_index = (i,j,dis)
                '''
                dis = np.linalg.norm(x-self.w[i,j])#欧式距离
                if dis < max_dis:
                    max_dis = dis
                    win_index = (i,j,dis)
                #'''
        return win_index

    def Get_Neighborhood(self, win, radius):
        res = []
        for i in range(max(0, win[0]-radius), min(self.w.shape[0], win[0]+radius+1)):
            for j in range(max(0, win[1]-radius), min(self.w.shape[1], win[1]+radius+1)):
                dis = (i-win[0])**2 + (j-win[1])**2
                if dis <= self.som_r_square: res.append([i, j, dis])
        return res

    def Update_W(self, index, X, r):
        for i in range(len(index)):
            self.w[index[i][0],index[i][1]] += self.l*self.h(index[i][2],r)*(X-self.w[index[i][0],index[i][1]])

    def Radius(self, t):
        return (self.som_r*(4.5/(t+3)))**2
    
    def alpha(self, t):
        if self.l <= 0.01: return 0.01
        else:              return 1./(2*t+1)

    def h(self, dis, r_square):
        return np.exp(-float(dis)/(2*r_square))
        
    def Train(self, fpath):
        self.in_layer = self.Normalize_Input(self.in_layer)
        for i in range(self.m_iter):
            p = int(46*i/self.m_iter)+1
            print(' P|'+'*'*p+' '*(46-p)+'| '+str(i)+'/'+str(self.m_iter), end='\r')
            self.w = self.Normalize_W(self.w)
            r = self.Radius(i)
            self.l = self.alpha(i)
            D = 0
            for k in range(self.in_layer.shape[0]):
                j = np.random.randint(self.in_layer.shape[0])
                win = self.Get_Win_Neuron(self.in_layer[j])
                index = self.Get_Neighborhood(win, self.som_r)
                self.Update_W(index, self.in_layer[j], r)
                D += win[2]
            self.D_list.append(D)
        self.Get_Result()
        print(' P|'+'*'*46+'| '+str(self.m_iter)+'/'+str(self.m_iter))
        return self.w.reshape(self.w.shape[0]*self.w.shape[1], self.w.shape[2])[list(self.neuron.keys())], self.neuron

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
                self.neuron[key]=[i]

    def Cluster(self):
        print('a')
    
    def Draw_Criterion(self):
        img,ax = plt.subplots(2,2,figsize=(10,7))
        t = np.arange(self.m_iter)
        y = np.array(self.D_list)
        x = self.D_list.index(max(y))
        print(x)
        ax[0][0].plot(t, y)
        ax[0][0].axhline(self.in_layer.shape[0], color='r')
        ax[0][1].axhline(self.in_layer.shape[0], color='r')
        ax[1][0].axhline(self.in_layer.shape[0], color='r')
        ax[1][1].axhline(self.in_layer.shape[0], color='r')
        ax[0][0].scatter(x, y[x], c='r', marker='*')
        ax[0][1].plot(t[20:50], y[20:50])
        ax[1][0].plot(t[50:100], y[50:100])
        ax[1][1].plot(t[100:self.m_iter], y[100:self.m_iter])
        img.savefig('D')
        plt.close()

    def Draw_Process(self, fpath):
        if not os.path.exists('./neuron-som-'+fpath): os.makedirs('./neuron-som-'+fpath)
        self.Get_Result()
        img =plt.figure(figsize=(10, 4.8))
        ax2 = img.add_subplot(122)
        if   self.w.shape[2]==2: ax1 = img.add_subplot(121)
        elif self.w.shape[2]==3: ax1 = img.add_subplot(121, projection='3d')
        for key in self.neuron.keys():
            a, b = int(key / self.w.shape[0]), key % self.w.shape[0]
            ax2.scatter(a, b, c=self.color[int(self.s[self.neuron[key][0]])], marker='.')
            if self.w.shape[2] == 2:
                ax1.scatter(self.w[a][b][0], self.w[a][b][1], c=self.color[int(self.s[self.neuron[key][0]])], marker='.')
            elif self.w.shape[2] == 3:
                ax1.scatter(self.w[a][b][0], self.w[a][b][1], self.w[a][b][2], c=self.color[int(self.s[self.neuron[key][0]])], marker='.')
        img.savefig('./neuron-som-'+fpath+'/pro-'+str(i))
        plt.close()

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
    else:             size = max(min(100, int(np.sqrt(dataset.shape[0])*2)), 10)

    if len(argv) > 3: iteration = int(argv[3])
    else:             iteration = 50

    print(' Dataset:', dataset.shape, '| Grid:', size, '| Iteration:', iteration)

    if dataset.shape[1]==dimension: dataset = np.hstack((dataset, np.zeros((dataset.shape[0], 1))))
    
    som = SOM(dataset[:, :dimension], dataset[:, -1], (size, size), iteration)
    #som.Init_W()
    out, res = som.Train(fname)
    som.Draw_Criterion()
    
    fig =plt.figure(figsize=(10, 9.5))
    ax12 = fig.add_subplot(222)
    ax22 = fig.add_subplot(224)
    som.Draw_SOM_Grid(ax12, ax22, str(len(res)))

    if dimension==2:
        ax11 = fig.add_subplot(221)
        ax21 = fig.add_subplot(223)
        for i in range(dataset.shape[0]):
            ax11.scatter(dataset[i,0], dataset[i,1], c=som.color[int(dataset[i,2])], marker='.')
            ax11.axis('off')
        for key in som.neuron.keys():
            i, j = int(key / som.w.shape[0]), key % som.w.shape[0]
            ax21.scatter(som.w[i][j][0], som.w[i][j][1], c=som.color[int(som.s[som.neuron[key][0]])], marker='.')
    elif dimension==3:
        ax11 = fig.add_subplot(221, projection='3d')
        ax21 = fig.add_subplot(223, projection='3d')
        ax11.view_init(azim=45)
        for i in range(dataset.shape[0]):
            ax11.scatter(dataset[i,0], dataset[i,1], dataset[i,2], c=som.color[int(dataset[i,3])], marker='.')
        for key in som.neuron.keys():
            i, j = int(key / som.w.shape[0]), key % som.w.shape[0]
            ax21.scatter(som.w[i][j][0], som.w[i][j][1], som.w[i][j][2], c=som.color[int(som.s[som.neuron[key][0]])], marker='.')
    ax11.set_title('source-'+str(dataset.shape[0]))
    ax21.set_title('neuron-'+str(len(res)))

    fig.savefig(fname+'-'+str(size)+'-'+str(iteration))
    plt.close()
    print(' Neurons:', len(res), '| Dataset:', len(dataset), '| Grid:', size, '| Iteration:', iteration)

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