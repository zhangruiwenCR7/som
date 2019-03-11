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
        self.label = []
        self.res =[]
        self.neuron = {}
        self.l = 0 
        self.s = s
        
    def Normaliza_Input(self, X):
        '''
        for i in range(X.shape[0]):
            t = np.linalg.norm(X[i])
            X[i] /= t
        #'''
        return X

    def Normaliza_W(self, w):
        '''
        for i in range(w.shape[0]):
            for j in range(w.shape[1]):
                t = np.linalg.norm(w[i,j])
                w[i,j] /= t
        #'''
        return w
    
    def Get_Win_Neuron(self, x):
        max_dis=100
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
        #print res
        return res
        #return [[win[0], win[1], 0]]

    def Update_W(self, index, time, X, r):
        for i in range(len(index)):
            self.w[index[i][0],index[i][1]] += self.Learning_Rate(index[i][2], time, r)*(X-self.w[index[i][0],index[i][1]])

    def Radius(self, t):
        #return 2*(1-t/self.m_iter)
        #return self.w.shape[0]*(1-t/self.m_iter)
        return int(self.w.shape[0]*(1-float(t)/self.m_iter)/2)+1

    def Learning_Rate(self, dis, time, r):
        #return np.exp(-dis)/(time+2)
        #self.l = 0.6*(1-float(time)/self.m_iter)*np.exp(-float(dis)/r)
        #print time, self.m_iter,dis, r, self.l
        self.l = 0.6*np.exp(1/(time-self.m_iter))*np.exp(-float(dis)/r)
        return self.l

    def Get_Result(self):
        self.w = self.Normaliza_W(self.w)
        #print self.w
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
        #print self.neuron
        
    def Train(self, learning_rate=1, threshold=1):
        self.in_layer = self.Normaliza_Input(self.in_layer)
        #print self.in_layer
        for i in range(self.m_iter):
            if i%50 == 0 and i >0: 
                print 'iteration:', i
                '''
                color = ['y', 'r', 'g', 'b']
                self.Get_Result()
                for key in self.neuron.keys():
                    k, j = key / self.w.shape[0], key % self.w.shape[0]
                    plt.scatter(k, j, c=color[int(self.s[self.neuron[key][0]])])
                    #print dataset[res[key][0], 2]
                    plt.savefig('2D-25-'+str(i))
                plt.close()
                '''
            #print self.l
            self.w = self.Normaliza_W(self.w)
            #print self.w
            for k in range(self.in_layer.shape[0]):
                j = np.random.randint(self.in_layer.shape[0])
                win = self.Get_Win_Neuron(self.in_layer[j])
                r = self.Radius(i)
                index = self.Get_Neighborhood(win, r)
                self.Update_W(index, i, self.in_layer[j], r)
        self.Get_Result()
        return self.w.reshape(self.w.shape[0]*self.w.shape[1], self.w.shape[2])[self.neuron.keys()], self.neuron
        #self.Preprocessing()
        '''
        self.res = self.Agglomeration()
        c=[]
        for i in range(len(self.label)):
            for j in range(len(self.res)):
                if self.label[i] in list(self.res[j]):
                    c.append(j)
        print c
        return np.array(c)
        '''
    
    def Preprocessing(self):
        dev =[]
        for key in self.neuron.keys():
            mean_x = np.mean(self.in_layer[self.neuron[key]])
            dev.append(np.linalg.norm(self.w[key/self.w.shape[0]][key%self.w.shape[1]]-mean_x, ord=1))
        mean_dev, std_dev = np.mean(dev), np.std(dev)
        j=0
        for i in range(len(dev)):
            if dev[i] > mean_dev + std_dev:
                del self.neuron[self.neuron.keys()[i-j]]
                j += 1

        num=[]
        for key in self.neuron.keys():
            dis = []
            for x in self.neuron[key]:
                dis.append(np.linalg.norm(self.w[key/self.w.shape[0]][key%self.w.shape[1]]-x, ord=1))
            mean_dis, std_dis = np.mean(dis), np.std(dis)
            j=0
            for i in range(len(dis)):
                if dis[i] > mean_dis + std_dis:
                    del self.neuron[key][i-j]
                    j+=1
            num.append(len(self.neuron[key]))
        mean_num, std_num = np.mean(num), np.std(num)
        print num, mean_num,std_num
        for i in range(len(num)):
            if num[i] < mean_num - std_num:
                del self.neuron[self.neuron.keys()[i-j]]
                j += 1
        print self.neuron

    def Agglomeration(self):
        res = []
        res.append(set([self.label[0]]))

        temp = -1
        for c in self.label:
            if temp == c: continue
            temp = c

            c_list = []
            print c
            '''
            index = -1
            for k in range(len(res)):
                if set([c]) & res[k] != set():
                    index = k
            if index == -1:
                res.append(set([c]))
                index = len(res)-1
            else:
                res[index] |= set([c])
            '''
            i, j = c / self.w.shape[0], c % self.w.shape[0]
            for x in range(max(i-1, 0), min(i+1, self.w.shape[0]-1)+1):
                for y in range(max(j-1, 0), min(j+1, self.w.shape[1]-1)+1):
                    c_list.append(x*self.w.shape[0] + y)
            r_list = list(set(self.label) & set(c_list))
            r_list.remove(c)
            #print r_list
            if r_list != None:
                for p in r_list:
                    x, y = p / self.w.shape[0], p % self.w.shape[0]
                    dis = np.linalg.norm(self.w[i][j] - self.w[x][y])
                    min_dis = 10000
                    if dis < min_dis:
                        la = set([x*self.w.shape[0] + y])
                        min_dis = dis
                print la
                index_c = index_la = -1
                for k in range(len(res)):
                    if set([c]) & res[k] != set():
                        index_c = k
                    if la & res[k] != set():
                        index_la = k
                if index_c == -1 and index_la == -1:
                    res.append(set([c]) | la)
                else:
                    if index_c != -1 and index_la == -1:
                        res[index_c] |= la
                    elif index_c == -1 and index_la != -1:
                        res[index_la] |= set([c])
                    elif index_c == index_la:
                        res[index_la] |= set([c]) | la
                    else:
                        res[index_la] |= res[index_c]
                        del res[index_c]
        print res
        return res
        #sys.exit()

def main(argv):
    #fname = 'C'
    fname = argv[0]
    size = int(argv[1])
    iteration = int(argv[2])
    dataset = np.loadtxt('./dataset/'+fname+'.txt')
    print dataset.shape
    plt.scatter(dataset[:,0], dataset[:,1], c=dataset[:,2])
    #plt.scatter(dataset[:,0], dataset[:,1])
    plt.title(fname+'dataset')
    plt.savefig(fname+'-dataset')
    plt.close()

    #som = SOM(dataset[:, :2], dataset[:, 2], (35,35), 30)
    som = SOM(dataset[:, :2], dataset[:, 2], (size, size), iteration)
    out, res = som.Train()
    print len(res)
    color = [0, 'y', 'r', 'g', 'b', 'c', 'm', 'k']
    for key in res.keys():
        i, j = key / som.w.shape[0], key % som.w.shape[0]
        plt.scatter(i, j, c=color[int(dataset[res[key][0], 2])])
        #print dataset[res[key][0], 2]
    plt.savefig(fname+'-som-2D')
    plt.close()

    '''
    som1 = SOM(out, (1,3), 1000)
    out1, res1 = som1.Train()
    print res1

    label = np.zeros(len(dataset))
    for key in res1.keys():
        for i in res1[key]:
            key1 = res.keys()[i]
            for j in res[key1]:
                label[j] = key
    print label

    
    plt.scatter(dataset[:,0], dataset[:,1], c=label)
    plt.title('result')
    plt.savefig('res')
    plt.close()

    plt.scatter(out[:,0], out[:,1], c=res.keys())
    plt.title('result')
    plt.savefig('res2')
    plt.close()
    '''

if __name__ == '__main__':
    print '\n ##--Start---- By Zhang Ruiwen ---', time.asctime( time.localtime(time.time()) ), '------\n' 
    main(sys.argv[1:])
    print '\n ##--End------ By Zhang Ruiwen ---', time.asctime( time.localtime(time.time()) ), '------\n' 
        
        
