#!/usr/bin/python3
#--- coding:utf-8
## @Copyright by DENG, Zhidong & Fu, Zhao, Dept of CS, Tsinghua Univsersity
## Updated on March 2, 2019
## Density-based Distance Tree (DDT)
import time, sys, os
from IPython import embed
import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from matplotlib import cm
import LCCV
import DBSCAN as ds

def main(argv):
    fname = argv[0]
    fig =plt.figure(figsize=(10, 9.5))
    data = np.loadtxt('/mnt/c/document/01research/01cluster/04dataset/'+fname+'.txt')
    if data.shape[1] > 2: 
        data = np.delete(data, 2, axis = 1)
    
    if fname == 'A' or fname == 'db3' or fname == 'sn':
        noise, _, _ = ds.dbscan(data, 3, [])
    elif fname == 'B' or fname == 'five_cluster' or fname=='sk':
        noise, _, _ = ds.dbscan(data, 9, [])
    elif fname == 'cth' or fname == 'fc1' or fname== 'three_cluster':
        noise, _, _ = ds.dbscan(data, 5, [])
    elif fname == 'D':
        noise, _, _ = ds.dbscan(data, 4, [])
    try:
        noise = noise.reshape(data.shape[0],)
        ind = np.argwhere(noise != -1)
        data = data[ind.reshape(len(ind),)]
    except:
        pass
    # minx, maxx = min(data[:,0]), max(data[:,0])
    # miny, maxy = min(data[:,1]), max(data[:,1])
    # data[:,0] = (data[:,0]-minx)/(maxx-minx)
    # data[:,1] = (data[:,1]-miny)/(maxy-miny)
    s = LCCV.LCCV(data)
    pwd = s.Dist_M()
    s.NaN()
    
    shortest_path = np.loadtxt('/mnt/c/document/01research/01cluster/04dataset/graphdistance_n/'+fname+'_Graph_Dis.txt')
    # mins, maxs = np.min(shortest_path), np.max(shortest_path)
    # shortest_path = (shortest_path-mins)/(maxs-mins)
    # cores = np.arange(0, data.shape[0], 1)
    # shortest_path = s.Graph_Dist(cores)
    # np.savetxt('/mnt/c/document/01research/01cluster/04dataset/graphdistance_n/'+fname+'_Graph_Dis.txt', shortest_path)
    rho, _, ordrho = s.Density_DPD(pwd)
    ord_gamma, parent, nchild, gamma, gamma_sorted, alpha = s.DPD(rho, ordrho, shortest_path)
    
    if len(argv)>1: 
        nc = int(argv[1])
        cl, peak_root = s.DPD_Cluster(nc, ordrho, ord_gamma, parent)
        childindex = s.Leaf_node(peak_root, parent, nchild)
        bestcl = cl
        x=np.arange(1,21)
        ax1=fig.add_subplot(221)
        # plt.plot(x,gamma_sorted[0:20])
        ax1.plot(x, alpha)
        ax1.set_xlim(0,21)
        # plt.ylim(0,2000)
        ax1.scatter(nc, alpha[nc-1], c='r', marker='o')
        ax1.set_title('alpha')
        # plt.savefig('./res_alpha/'+fname+'-gamma')
        # plt.close()

        g = np.zeros((18,1))
        for i in range(18):
            g[i] = (gamma_sorted[i] - gamma_sorted[i+1])-(gamma_sorted[i+1] - gamma_sorted[i+2])
            # g[i] = np.arctan(gamma_sorted[i] - gamma_sorted[i+1])+np.arctan(1/(gamma_sorted[i+1] - gamma_sorted[i+2]))
            # print(np.arctan(gamma_sorted[i] - gamma_sorted[i+1])*180/3.14, np.arctan(1/(gamma_sorted[i+1] - gamma_sorted[i+2]))*180/3.14)
        ind = np.argmin(g)
        x1 = np.arange(2,20)
        ax2=fig.add_subplot(222)
        ax2.plot(x1, g)
        ax2.set_xlim(0,20)
        ax2.scatter(nc, g[nc-2], c='r', marker='o')
        ax2.scatter(ind+2, g[ind], c='g', marker='x')
        ax2.set_title('gamma-diff')
        # plt.savefig('./res_gamma_diff/'+fname+'-gamma-diff')
        # plt.close()

    else:
        total_iter = 21
        lccv_indx = np.zeros((total_iter, 1))
        lccv_indx1 = np.zeros((total_iter, 1))
        lccv_indx2 = np.zeros((total_iter, 1))
        maxlccv_indx = -1
        for nc in range(2, total_iter):
            cl, peak_root = s.DPD_Cluster(nc, ordrho, ord_gamma, parent)
            childindex = s.Leaf_node(peak_root, parent, nchild)
            lccv_indx[nc], lccv_indx1[nc], lccv_indx2[nc] = s.ComputeVI(peak_root, cl, childindex, shortest_path, rho)
            #lccv_indx[nc] = s.ComputeLCCV(cl, cores, shortest_path, cores)
            print (fname, 'nc = %d   ncl = %d   LCCV = %f' % (nc, max(cl), lccv_indx[nc]))
            if maxlccv_indx < lccv_indx[nc]:
                maxlccv_indx = lccv_indx[nc]
                bestcl = cl
        min1, max1 = min(lccv_indx1[2:]), max(lccv_indx1[2:])
        lccv_indx1 = (lccv_indx1-min1)/(max1-min1)
        min2, max2 = min(lccv_indx2[2:]), max(lccv_indx2[2:])
        lccv_indx2 = (lccv_indx2-min2)/(max2-min2)
        VI = lccv_indx1 *lccv_indx2
        print(np.argmax(VI))
        ax2=fig.add_subplot(223)
        x=np.arange(2,20)
        ax2.plot(x, lccv_indx2[2:20], c='g')
        ax22=ax2.twinx()
        ax22.plot(x, lccv_indx1[2:20], c='r')
        ax2.set_xlim(0,21)
        ax2.set_title('IV')
        ax3=fig.add_subplot(222)
        ax3.plot(x, VI[2:20])
        ax3.set_xlim(0,21)
        # plt.ylim(-1, maxlccv_indx)
        # plt.savefig('./VI/'+fname+'-VI')
        # plt.close()
    # classes = bestcl.copy()
    classes, _ = s.DPD_Cluster(np.argmax(VI), ordrho, ord_gamma, parent)
    nclust = int(np.max(classes))
    
    '''
    s.Density_LORE()
    cl, cores, local_core = s.LORE()
    shortest_path = s.Graph_Dist(cores)
    print(cores, local_core)
    total_iter = int(np.sqrt(data.shape[0]))+1
    lccv_indx = np.zeros((total_iter, 1))
    maxlccv_indx = -1
    for nc in range(2, 3):
        cl, p_type, no = ds.dbscan(data, nc, [])
        cl = cl.transpose()
        for i in range(s.N):
            if (cl[i] != -1) and (cl[i] != cl[local_core[i]]):
                cl[i] = cl[local_core[i]]
        for i in range(int(np.max(cl))):
            temp1 = np.argwhere(cl == (i + 1))
            if len(temp1) == 0:
                for j in range(i, int(np.max(cl))):
                    temp2 = np.argwhere(cl == (j + 1))
                    if len(temp2) != 0:
                        for k in range(len(temp2)):
                            cl[temp2[k]] = cl[temp2[k]] - 1
        lccv_indx[nc] = s.ComputeLCCV(cl, cores, shortest_path, local_core)
        print ('nc = %d   ncl = %d   LCCV = %f' % (nc, max(cl), lccv_indx[nc]))
        if maxlccv_indx < lccv_indx[nc]:
            maxlccv_indx = lccv_indx[nc]
            bestcl = cl
    print ('Process ends')
    classes = bestcl.copy()
    nclust = int(np.max(bestcl))
    #'''
    #--------------------------------------------------------------
    ax4=fig.add_subplot(224)
    start = 0.2
    stop = 0.8
    cm_subsection = np.linspace(start, stop, nclust)
    colors = [cm.jet(x) for x in cm_subsection]
    # use colors as indicators of different categories
    for i in range(data.shape[0]):
        if classes[i] > 0:
            ic = int(classes[i]) - 1
            ax4.plot(data[i, 0], data[i, 1], marker='.', markerfacecolor=colors[ic], markersize=8, markeredgecolor=colors[ic])
        else:
            ax4.plot(data[i, 0], data[i, 1], marker='.', markerfacecolor='k', markersize=8, markeredgecolor='k')
    # draw every tree-like cluster
    for k in range(nclust):
        for i in range(nclust, data.shape[0]):
            if (classes[parent[ordrho[i]]] == classes[ordrho[i]]) and (classes[ordrho[i]] == (k + 1)):
                xx = [data[parent[ordrho[i]], 0], data[ordrho[i], 0]]
                yy = [data[parent[ordrho[i]], 1], data[ordrho[i], 1]]
                ax4.plot(xx, yy, linestyle='-', linewidth=2, color="#999999")
    # draw a relational tree among cluster heads
    for k in range(nclust - 1):
        npeak0 = int(peak_root[k, data.shape[1]])
        npeak1 = int(peak_root[k + 1, data.shape[1]])
        xx = [data[npeak1, 0], data[npeak0, 0]]
        yy = [data[npeak1, 1], data[npeak0, 1]]
        ax4.plot(xx, yy, linestyle='-', linewidth=2.5, color="#000000")
    # the density peak as root node of a tree-like cluster
    ax4.plot(peak_root[0, 0], peak_root[0, 1], marker='o', markersize=10, markerfacecolor="#FF0000", markeredgecolor='k')
    for i in range(1, nclust):
        ax4.plot(peak_root[i, 0], peak_root[i, 1], marker='o', markersize=10, markerfacecolor="#BBBBBB", markeredgecolor='r')
    # for i in range(len(childindex)):
    #     ax4.plot(data[childindex[i], 0], data[childindex[i], 1], marker='x', markersize=10, markerfacecolor="#BBBBBB", markeredgecolor='r')
    # fig.savefig('./VI31/'+fname)
    fig.savefig(fname)
    # plt.close()
    #Hot(data, rho, 1, fname)
#data：样本
#rho: 密度
#sigma: 正太分布方差
def Hot(data, rho, sigma, fname):
    N, dim = data.shape
    xmin, xmax = min(data[:,0]), max(data[:,0])
    ymin, ymax = min(data[:,1]), max(data[:,1])
    x = np.linspace(xmin-(xmax-xmin)/10, xmax+(xmax-xmin)/10, N)
    y = np.linspace(ymin-(ymax-ymin)/10, ymax+(ymax-ymin)/10, N) #生成X，Y的队列
    z = np.zeros((len(x), len(x)))
    for k in range(N):
        X=np.repeat((x-data[k,0])**2,len(x)).reshape(len(x),len(x))
        Y=np.repeat((y-data[k,1])**2,len(x)).reshape(len(x),len(x)).T
        z += rho[k]*np.exp((-Y-X)/2/sigma**2)
    X,Y=np.meshgrid(x,y) #生成X，Y网格
    cmap = plt.cm.get_cmap("rainbow") #选择rainbow色彩
    plt.contourf(X, Y, z.T, 20, alpha = 0.8, cmap = cmap) #色彩分为20级
    plt.xticks([])
    plt.yticks([])
    cb = plt.colorbar() #绘制色彩图注
    cb.set_label('rho')
    plt.savefig(fname+'hot') #保存图片

    # X = np.linspace(min(data[:,0])-1, max(data[:,0])+1, s.N)
    # Y = np.linspace(min(data[:,1])-1, max(data[:,1])+1, s.N)
    # Height = fun(data, rho, X, Y, pwd)
    # X,Y=np.meshgrid(X,Y)
    # cmap = plt.cm.get_cmap("rainbow")
    # plt.contourf(X, Y, Height, 20, alpha = 0.8, cmap = cmap)
    # plt.xticks([])
    # plt.yticks([])
    # cb = plt.colorbar()
    # cb.set_label('rho')
    # plt.savefig(fname+'hot')

def fun(data,rho,x,y,pwd):
    z=np.zeros((len(x), len(x)))
    pwd = sorted(pwd)
    l = int(len(x))
    sigma = pwd[l]
    # sigma = np.mean(pwd)
    # sigma = 5
    print(l,sigma)
    # for i in range(len(x)):
    #     for j in range((len(y))):
    #         for k in range(data.shape[0]):
    #             sigma = 1
    #             z[j,i] += rho[k]*np.exp(-( (x[i]-data[k,0])**2+(y[j]-data[k,1])**2 )/2/sigma**2)
        
    for k in range(data.shape[0]):
        X=np.repeat((x-data[k,0])**2,len(x)).reshape(len(x),len(x))
        Y=np.repeat((y-data[k,1])**2,len(x)).reshape(len(x),len(x)).T
        z += rho[k]*np.exp((-Y-X)/2/sigma**2)
    return z.T

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
