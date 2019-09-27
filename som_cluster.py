#!/usr/bin/python3
#--- coding:utf-8

import time, sys, os
from IPython import embed
import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class Kmeans():
    def __init__(self, n_clusters, max_iter = 1000, tol = 0.00001):
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

class SOM():
    def __init__(self, n_clusters, in_layer, s, knn=10, out_size=(3,3), m_iter=1000):
        self.n_clusters = n_clusters
        self.in_layer = in_layer.copy()
        self.m_iter = m_iter
        self.knn = knn
        a,b = np.min(self.in_layer), np.max(self.in_layer)
        self.w = (b-a)*np.random.rand(out_size[0], out_size[1], self.in_layer.shape[1])+a
        self.color = ['y', 'r', 'g', 'b', 'c', 'm', 'k', 'pink', 'dark', 'orange', 'tan', 'gold']
        self.label = np.zeros(len(in_layer))
        self.res =None
        self.neuron = {}
        self.l = 1.0 
        self.s = s
        self.som_r = int(self.w.shape[0]/2.5)
        self.som_r_square = self.som_r**2
        self.D_list = []
    
    def Init_W(self):
        for i in range(self.w.shape[0]):
            for j in range(self.w.shape[1]):
                k = np.random.randint(self.in_layer.shape[0])
                self.w[i][j]=self.in_layer[k]
        
    def Normalize_Input(self, X):
        '''
        for i in range(X.shape[0]):
            t = np.linalg.norm(X[i])
            X[i] /= t
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
        max_dis=float('inf')
        min_dis=-float('inf')
        for i in range(self.w.shape[0]):
            for j in range(self.w.shape[1]):
                '''
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

    def Get_Result(self):
        self.w = self.Normalize_W(self.w)
        self.neuron={}
        for i in range(self.in_layer.shape[0]):
            win = self.Get_Win_Neuron(self.in_layer[i])
            key = win[0]*self.w.shape[0] + win[1]
            if key in self.neuron.keys():
                self.neuron[key].append(i)
            else:
                self.neuron.fromkeys([key])
                self.neuron[key]=[i]
                  
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

    def Preprocess(self):
        #filter outliers
        dev=[]
        key_l = list(self.neuron.keys())
        for key in key_l:
            a, b=int(key/self.w.shape[0]), key%self.w.shape[0]
            mean_x = self.in_layer[self.neuron[key]].mean(axis=0)
            d=np.sum((self.w[a][b]-mean_x)**2)
            dev.append(d)
        mean_dev, std_dev=np.mean(dev), np.std(dev)
        for i in range(len(dev)):
            if dev[i] > mean_dev+std_dev:
                del self.neuron[key_l[i]]
        #embed(header='First time')
        #filter outliers and noises
        dev=[]
        key_l = list(self.neuron.keys())
        for key in key_l:
            for v in self.neuron[key]:
                a, b=int(key/self.w.shape[0]), key%self.w.shape[0]
                d=np.sum((self.in_layer[v]-self.w[a][b])**2)
                dev.append(d)
        mean_dev, std_dev=np.mean(dev), np.std(dev)
        cnt=0
        for key in key_l:
            for v in self.neuron[key]:
                if dev[cnt] > mean_dev+std_dev:
                    self.neuron[key].remove(v)
                cnt+=1
            if self.neuron[key] == '':
                del self.neuron[key]
        #filter noises
        dev=[]
        key_l = list(self.neuron.keys())
        for key in key_l:
            a,b = int(key/self.w.shape[0]),key%self.w.shape[0]
            temp=[]
            radius=1
            while len(temp)==0:
                for i in range(max(0, a-radius), min(self.w.shape[0], a+radius+1)):
                    for j in range(max(0, b-radius), min(self.w.shape[1], b+radius+1)):
                        #if i*self.w.shape[0]+j in key_l:
                        temp.append(np.sum((self.w[a][b]-self.w[i][j])**2))
                radius+=1
            dev.append(np.mean(temp))
        mean_dev,std_dev = np.mean(dev), np.std(dev)
        for i in range(len(dev)):
            if dev[i]>mean_dev+std_dev*3:
                del self.neuron[key_l[i]]
            
        t=np.arange(len(dev))
        dev.sort()
        plt.plot(t, np.array(dev))
        plt.axhline(mean_dev, color='r')
        plt.axhline(mean_dev+std_dev, color='g')
        plt.axhline(mean_dev+3*std_dev, color='b')
        plt.savefig('dev')
        #embed()
        '''
        dev=[]
        key_l = list(self.neuron.keys())
        for key in key_l:
            dev.append(len(self.neuron[key]))
        mean_dev, std_dev=np.mean(dev), np.std(dev)
        for key in key_l:
            if len(self.neuron[key])<mean_dev-std_dev:
                del self.neuron[key]
        '''

    def Get_Dis(self, X):
        S = np.zeros((len(X), len(X)))
        for i in range(len(X)):
            a,b=int(X[i]/self.w.shape[0]), X[i]%self.w.shape[0]
            for j in range(i+1, len(X)):
                c,d=int(X[j]/self.w.shape[0]), X[j]%self.w.shape[0]
                #a,b=int(X[i]/self.w.shape[0])-int(X[j]/self.w.shape[0]), X[i]%self.w.shape[0]-X[j]%self.w.shape[0]
                #if abs(a-c)<self.knn and abs(b-d)<self.knn:
                if (a-c)**2+(b-d)**2<self.knn**2:
                    S[i][j] = 1
                    S[j][i] = S[i][j]
        return S

    def Get_W_KNN(self, X, S):
        W = np.zeros((len(X), len(X)))
        for i in range(len(X)):
            #index = np.argpartition(S[i], self.knn)[:self.knn+1]
            index = np.argwhere(S[i]>0).flatten()
            a,b=int(X[i]/self.w.shape[0]), X[i]%self.w.shape[0]
            if len(index) < self.knn:
                dis=np.zeros(len(X))
                for j in range(len(X)):
                    c,d=int(X[j]/self.w.shape[0]), X[j]%self.w.shape[0]
                    dis[j] = np.sum((self.w[a][b]-self.w[c][d])**2)
                index = np.argpartition(dis, self.knn)[:self.knn]
                W[i,index]= np.exp(-dis[index]/self.sigma)
            else:
                for j in index:
                    c,d=int(X[j]/self.w.shape[0]), X[j]%self.w.shape[0]
                    temp = np.sum((self.w[a][b]-self.w[c][d])**2)
                    W[i,j]= np.exp(-temp/self.sigma)
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
        from sklearn.cluster import KMeans
        self.res = KMeans(self.n_clusters).fit(vec).labels_+1

    def Get_Labels(self):
        cnt=0
        for key in self.neuron.keys():
            for ind in self.neuron[key]:
                self.label[ind] = self.res[cnt]
            cnt+=1

    def Cluster(self):
        #self.Preprocess()
        self.sigma = 1.
        self.fit(list(self.neuron.keys()))
        self.Get_Labels()

    def Draw_Criterion(self):
        img,ax = plt.subplots(2,2,figsize=(10,7))
        t = np.arange(self.m_iter)
        y = np.array(self.D_list)
        x = self.D_list.index(max(y))
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

    def Draw_SOM_Grid(self, fig, location=222):
        ax = fig.add_subplot(location)
        cnt=0
        for key in self.neuron.keys():
            i, j = key / self.w.shape[0], key % self.w.shape[0]
            #f0.scatter(i, j, c=self.color[int(self.s[self.neuron[key][0]])], marker='.')
            ax.scatter(i, j, c=self.color[self.res[cnt]], marker='.')
            #f1.scatter(i, j, c='b', marker='.')
            cnt+=1
        ax.set_title('som-2D-'+str(self.w.shape[0]))
        #f0.axis('off')
        #f1.set_title('som-2D-'+str(self.w.shape[0]))

    def Draw_Neuron(self, fig, location=224):
        cnt=0
        if self.in_layer.shape[1]==2:
            ax = fig.add_subplot(location)
            for key in self.neuron.keys():
                i, j = int(key / self.w.shape[0]), key % self.w.shape[0]
                ax.scatter(self.w[i][j][0], self.w[i][j][1], c=self.color[self.res[cnt]], marker='.')
                cnt+=1
        elif self.in_layer.shape[1]==3:
            ax = fig.add_subplot(location, projection='3d')
            ax.view_init(azim=45)
            for key in self.neuron.keys():
                i, j = int(key / self.w.shape[0]), key % self.w.shape[0]
                ax.scatter(self.w[i][j][0], self.w[i][j][1], self.w[i][j][2], c=self.color[self.res[cnt]], marker='.')
                cnt+=1
        ax.set_title('neuron-'+str(len(self.neuron.keys())))

    def Draw_Clustering(self, fig, location=223):
        ax=fig.add_subplot(location)
        for i in range(self.in_layer.shape[0]):
            ax.scatter(self.in_layer[i,0], self.in_layer[i,1], c=self.color[int(self.label[i])], marker='.')
        ax.axis('off')
        ax.set_title('clustering-res')

    def Draw_Normalize(self):
        plt.scatter(self.in_layer[:,0], self.in_layer[:,1], c=np.zeros(len(self.s)))
        plt.scatter(self.in_layer[:,0], self.in_layer[:,1], c=self.s)
        plt.savefig('t')
        plt.close()
        print(' Image saved OK!')

def main(argv):
    fname = argv[0]
    dataset = np.loadtxt('../04dataset/'+fname+'.txt')

    if len(argv) > 1: n_cluster = int(argv[1])
    else:             dimension = 2

    if len(argv) > 2: knn = int(argv[2])
    else:             knn = None

    if len(argv) > 3: iteration = int(argv[3])
    else:             iteration = 100

    if len(argv) > 4: size = int(argv[4])
    else:             
        #size = max(min(100, int(np.sqrt(dataset.shape[0])*2)), 10)
        size = int(np.sqrt(dataset.shape[0]))

    print(' Dataset:', dataset.shape, '| Grid:', size, '| Iteration:', iteration)
    
    som = SOM(n_cluster, dataset[:, :dataset.shape[1]-1], dataset[:, -1], knn, (size, size), iteration)
    #som.Init_W()
    out, res = som.Train(fname)
    som.Cluster()
    #som.Draw_Criterion()
    
    fig =plt.figure(figsize=(10, 9.5))
    som.Draw_SOM_Grid(fig, 222)
    som.Draw_Clustering(fig, 223)
    som.Draw_Neuron(fig, 224)

    if dataset.shape[1]-1==2:
        ax11 = fig.add_subplot(221)
        for i in range(dataset.shape[0]):
            ax11.scatter(dataset[i,0], dataset[i,1], c=som.color[int(dataset[i,2])], marker='.')    
    elif dataset.shape[1]-1==3:
        ax11 = fig.add_subplot(221, projection='3d')
        ax11.view_init(azim=45)
        for i in range(dataset.shape[0]):
            ax11.scatter(dataset[i,0], dataset[i,1], dataset[i,2], c=som.color[int(dataset[i,3])], marker='.')
    ax11.axis('off')
    ax11.set_title(fname+'-source-'+str(dataset.shape[0]))

    fig.savefig(fname+'-'+str(size)+'-'+str(iteration))
    plt.close()
    print(' Neurons:', len(res), '| Dataset:', len(dataset), '| Grid:', size, '| Iteration:', iteration)

if __name__ == '__main__':
    start_second = time.time()
    print('\n ##--Start-- By Ruiwen --', time.asctime(time.localtime(start_second)), '------#\n')
    main(sys.argv[1:])
    run_second = time.time() - start_second
    s, m = int(run_second % 60), int(run_second / 60)
    rt = str(m)+':'+str(s)
    print('\n ##--End---- By Ruiwen --', time.asctime(time.localtime(time.time())), '--', rt, 's--#\n')