#!/usr/bin/python
#--- coding:utf-8
'''
    1.SOM(2D)
    2.Preprocess
    3.Merging
    4.Optimal

1.SOM
    input: X shape(N, D)
    output: W shape(a, b, D)
    Normaliza(X)
    Initial(W)
    for num_iter in max_iter:
        for x in X:
            for w in W:
                calculate_distance(x, w)
            get_win_neuron()
            update_W(num_iter, learning_rate, radius)
        if ||W - W_old|| < threshold:
            break
    add_label_to_X
    add_num_to_W

2.Preprocessing
    for w in W_1:
        mean = mean(x) #x belong to w 
        dev.append(||w - mean||)
    mean_dev = np.mean(dev)
    std_dev = np.std(dev)
    for i in len(W_1):
        if dev[i] > mean_dev+std_dev:
            change W_num[i] to 0

    for k in len(W_2):
        dis=[]
        for i in W_num:
            dis.append(||X[i] - w||)
        mean_dis[k] = np.mean(dis)
        std_dis[k] = np.std(dis)
        for i in W_num:
            if dis[i] > mean_dis[k]+std_dis[k]:
                delete X[i]

    mean_num = np.mean(W_2[:,:,0])
    std_num  = np.std(W_2[:,:,1])
    for w in W_2:
        if w[0] < mean_num - std_num:
            change w[0] to 0

3.CDbw
    for i in range(c):
        std_ev_p[i] = np.sqrt(np.sum(np.power((x_k-mean[i]),2)/(W_num[i]-1)))
    std_ev = np.sqrt(np.sum(np.linalg.norm(std_ev_p)/c))

    #intra_distance
    for i in range(c):
        for x in X[i]:
            if np.linalg.norm(x-w[i]) < std_ev:
                temp += 1
    intra_den = temp/c

    #inter_distance
    for i in range(c):       

'''

import time
import sys
import matplotlib
matplotlib.use('Agg')
import numpy as np
from matplotlib import pyplot as plt

class SOM():
    def __init__(self, in_layer, out_size=(3,3), m_iter=1000):
        self.in_layer = in_layer.copy()
        self.m_iter = m_iter
        self.w = np.random.rand(out_size[0], out_size[1], self.in_layer.shape[1])
        self.label = []
        self.res =[]
        
    def Normaliza_Input(self, X):
        '''
        for i in range(X.shape[0]):
            t = np.linalg.norm(X[i])
            X[i] /= t
        '''
        return X

    def Normaliza_W(self, w):
        '''
        for i in range(w.shape[0]):
            for j in range(w.shape[1]):
                t = np.linalg.norm(w[i,j])
                w[i,j] /= t
        '''
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
                dis = np.linalg.norm([i-win[0], j-win[1]])
                if dis < radius:
                    res.append([i, j, dis])
        #print res
        return res
        #return [[win[0], win[1], 0]]

    def Update_W(self, index, time, X):
        for i in range(len(index)):
            self.w[index[i][0],index[i][1]] += self.Learning_Rate(index[i][2], time)*(X-self.w[index[i][0],index[i][1]])

    def Radius(self, t):
        return 2*(1-t/self.m_iter)
        #return self.w.shape[0]*(1-t/self.m_iter)

    def Learning_Rate(self, dis, time):
        return np.exp(-dis)/(time+2)

    def Get_Result(self):
        self.w = self.Normaliza_W(self.w)
        #print self.w
        for i in range(self.in_layer.shape[0]):
            win = self.Get_Win_Neuron(self.in_layer[i])
            key = win[0]*self.w.shape[0] + win[1]
            self.label.append(key)
        '''
        print key,i
        if label.has_key(key):
            label[key].append(i)
        else:
            label.fromkeys([key])
            label[key]=[]
            label[key].append(i)
        '''
        
    def Train(self, learning_rate=1, threshold=1):
        self.in_layer = self.Normaliza_Input(self.in_layer)
        #print self.in_layer
        for i in range(self.m_iter):
            self.w = self.Normaliza_W(self.w)
            #print self.w
            for j in range(self.in_layer.shape[0]):
                #j = np.random.randint(self.in_layer.shape[0])
                win = self.Get_Win_Neuron(self.in_layer[j])
                r = self.Radius(i)
                index = self.Get_Neighborhood(win, r)
                self.Update_W(index, i, self.in_layer[j])
        self.Get_Result()
        self.res = self.Agglomeration()
        c=[]
        for i in range(len(self.label)):
            for j in range(len(self.res)):
                if self.label[i] in list(self.res[j]):
                    c.append(j)
        print c
        return np.array(c)

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
    dataset = np.loadtxt('./E.txt')
    print dataset.shape, dataset[0]
    #plt.scatter(dataset[:,0], dataset[:,1], c=dataset[:,2])
    plt.scatter(dataset[:,0], dataset[:,1])
    plt.title('testdataset')
    plt.savefig('testdata')
    plt.close()

    som = SOM(dataset[:, :2], (10,10), 10)
    res = som.Train(0.9, 0.0001)
    print len(res)
    
    plt.scatter(dataset[:,0], dataset[:,1], c=res)
    plt.title('result')
    plt.savefig('res')
    plt.close()

    plt.scatter(som.in_layer[:,0], som.in_layer[:,1], c=res)
    plt.title('result')
    plt.savefig('res2')
    plt.close()
    #'''



if __name__ == '__main__':
    print '\n ##--Start---- By Zhang Ruiwen ---', time.asctime( time.localtime(time.time()) ), '------\n' 
    main(sys.argv[1:])
    print '\n ##--End------ By Zhang Ruiwen ---', time.asctime( time.localtime(time.time()) ), '------\n' 
        
        
