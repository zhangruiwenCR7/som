#!/usr/bin/python3
#--- coding:utf-8
## @Copyright by DENG, Zhidong, Department of Computer Science, Tsinghua University
## Updated on April 6, 2019
## Density-based Distance Tree (DDT) + LCCV
import time, sys, os
from IPython import embed
import numpy as np
import networkx as nx

class LCCV():
    def __init__(self, X):
        self.A = X
        self.N, self.dim = X.shape

    def Dist_M(self):
        pwd = np.zeros((int(self.N*(self.N-1)/2), 1))
        pwd_cnt = 0
        self.dist = np.zeros((self.N, self.N))
        for i in range(self.N):
            for j in range(i+1, self.N):
                pwd[pwd_cnt] = np.linalg.norm(self.A[i,:]-self.A[j,:])
                self.dist[i, j] = pwd[pwd_cnt]
                self.dist[j, i] = pwd[pwd_cnt]
                pwd_cnt += 1
        self.sdist = np.sort(self.dist, axis = 1)
        self.index = np.argsort(self.dist, axis = 1)
        return pwd

    def NaN(self):
        r = 1
        flag = cnt = cnt1 = 0
        self.nb = np.zeros((self.N, 1), dtype=np.int)
        while flag==0:
            for i in range(self.N):
                k = self.index[i, r]
                self.nb[k] += 1
            r += 1
            cnt2 = self.N-len(np.nonzero(self.nb)[0])#test
            if cnt1==cnt2: cnt += 1
            else:          cnt  = 1

            if cnt2==0 or (r>2 and cnt>=2): flag = 1
            cnt1 = cnt2
        self.supk = r - 1
        self.max_nb = np.max(self.nb)
        print (' The characteristic value is %d' % self.supk)
        print (' The maximum value of nb is %d' % self.max_nb)

    def Density_LORE(self):
        self.rho = np.zeros((self.N, 1))
        for i in range(self.N):
            self.rho[i] = self.max_nb/np.sum(self.sdist[i, :self.max_nb+1])
        self.rho_sorted = np.array(sorted(self.rho, reverse = True))
        self.ordrho = np.lexsort((self.rho_sorted, self.rho), axis = 0)[::-1]
    
    def LORE(self):
        local_core = -np.ones((self.N, 1), dtype=np.int)
        for i in range(self.N):
            p = self.ordrho[i]
            maxrho = self.rho[p]
            maxindex = p
            for j in range(int(self.nb[p])+1):
                x = self.index[p, j]
                if maxrho < self.rho[x]:
                    maxrho = self.rho[x]
                    maxindex = x
            if local_core[maxindex]==-1:
                local_core[maxindex] = maxindex
            for j in range(int(self.nb[p])+1):
                if local_core[self.index[p, j]] == -1:
                    local_core[self.index[p, j]] = local_core[maxindex]
                else:
                    if self.dist[self.index[p, j], local_core[self.index[p, j]]]>self.dist[self.index[p, j], local_core[maxindex]]:
                        local_core[self.index[p, j]] = local_core[maxindex]
                for k in range(self.N):
                    if local_core[k] == self.index[p, j]:
                        local_core[k] = local_core[self.index[p, j]]
        init_cluster_N = 0
        cores0 = -np.ones((self.N, 1), dtype=np.int)
        cl = np.zeros((self.N, 1), dtype=np.int)
        for i in range(self.N):
            if local_core[i] == i:
                cores0[init_cluster_N] = i
                init_cluster_N += 1
                cl[i] = init_cluster_N
        cores = cores0[:init_cluster_N]
        for i in range(self.N):
            cl[i] = cl[local_core[i]]
        print ('The number of initial clusters is %d' % init_cluster_N)
        return cl, cores, local_core

    def Graph_Dist(self, nodes, node_num=2):
        node_num = nodes.shape[0]
        weight = np.zeros((self.N, self.N))
        for i in range(self.N):
            for j in range(1, self.supk+1):
                weight[i, self.index[i,j]] = self.dist[i, self.index[i,j]]
        shortest_path = np.zeros((node_num,node_num))
        G = nx.DiGraph()
        for i in range(self.N):
            G.add_node(i)
            for j in range(self.N):
                if weight[i, j]==0: weight[i,j]= float('inf')
                G.add_weighted_edges_from([(i, j, weight[i, j])])
        for i in range(node_num):
            Dict = nx.single_source_dijkstra_path_length(G, int(nodes[i]))
            D = list(Dict.values())
            D_index = list(Dict.keys())
            for j in range(i+1, node_num):
                indx_ = D_index.index(nodes[j])
                shortest_path[i, j] = D[indx_]
                if shortest_path[i, j] == float('inf'): shortest_path[i, j] = 0
                shortest_path[j ,i] = shortest_path[i, j]
        maxd = np.max(shortest_path)
        for i in range(node_num):
            for j in range(i+1, node_num):
                if shortest_path[i, j] == 0:
                    shortest_path[i, j] = shortest_path[j, i] = maxd 
        return shortest_path

    def ComputeLCCV(self, cl, cores, shortpath, local_core):
        D = np.zeros((cores.shape[0], self.dim))
        cl_cores = np.zeros((cores.shape[0], 1), dtype=np.int)
        nl = np.zeros((cores.shape[0], 1), dtype=np.int)
        for i in range(cores.shape[0]):
            D[i, :] = self.A[cores[i], :]
            cl_cores[i] = cl[cores[i]]
        
        for i in range(cores.shape[0]):
            for j in range(self.N):
                if local_core[j] == cores[i] and cl[j]>=0:
                    nl[i] += 1
        cl_num = int(np.max(cl))
        s = self.ComputeSWC(D, cl_cores, cl_num, shortpath)
        lccv = np.sum(s*nl) / self.N
        return lccv

    def ComputeSWC(self, D, cl_cores, ncl, shortpath):
        n, d = D.shape
        cdata = -np.ones((ncl, n*d+1))
        cindex = np.zeros((ncl, n), dtype=np.int)
        for i in range(ncl):
            nump = 0
            for j in range(n):
                if cl_cores[j] == i+1:
                    for k in range(d):
                        cdata[i, nump*d+k] = D[j, k]
                    cindex[i, nump] = j
                    nump += 1
            cdata[i, n*d] = nump

        s1 = np.zeros((n, 1))
        for i in range(ncl):
            np1 = int(cdata[i, n*d])
            for j in range(1, np1):
                suma = 0
                for k in range(np1):
                    suma += shortpath[cindex[i, j], cindex[i, k]]
                aa = suma / (np1-1)
                bb = float('inf')
                for k in range(ncl):
                    np2 = int(cdata[k, n*d])
                    if k != i and np2 != 0:
                        sumd = 0
                        for l in range(np2):
                            sumd += shortpath[cindex[i,j], cindex[k,l]]
                        if bb > sumd/np2: bb = sumd/np2
                if max(aa, bb) != 0: s1[cindex[i,j]] = (bb-aa) / max(aa, bb)
                else:                s1[cindex[i,j]] = 0
        return s1

    def Kruskal(self):
        lowcost = self.dist[0, :]
        tempcost = np.zeros((self.N, 1))
        treeEdge = np.zeros((self.N - 1, 1))
        maxd = np.max(self.dist)
        for i in range(self.N-1):
            nexte = 0
            for j in range(self.N):
                if lowcost[j] == 0:
                    tempcost[j] = maxd
                else :
                    tempcost[j] = lowcost[j]
            nextcost = np.min(tempcost)
            nexte = np.argmin(tempcost)
            treeEdge[i] = nextcost
            for j in range(self.N):
                temp = self.dist[nexte, j]
                if (temp != 0.) and (temp <= lowcost[j]):
                    lowcost[j] = temp

    def Density_DPD(self, pwd):
        degree = np.zeros((self.N, self.N))
        percentage = np.arange(1.0, 5.1, 0.1)
        sorted_pwd = np.sort(pwd, axis = 0)
        for i in range(len(percentage)):
            position = int(round(self.N*(self.N-1)*percentage[i]/200.))
            dc = sorted_pwd[position]
            degree += np.exp(-(self.dist / dc) * (self.dist / dc))
        degree /= len(percentage)
        rho = np.sum(degree, axis = 0)
        rho_sorted = np.array(sorted(rho, reverse = True))
        ordrho = np.lexsort((rho_sorted, rho))[::-1]
        # minr, maxr = np.min(rho), np.max(rho)
        # rho = (rho-minr)/(maxr-minr)
        return rho, rho_sorted, ordrho

    def DPD(self, rho, ordrho, dist):
        delta = np.zeros((self.N, 1))
        parent = np.zeros((self.N, 1), dtype=np.int)
        nchild = np.zeros((self.N, 1), dtype=np.int)
        delta[ordrho[0]] = -1
        parent[ordrho[0]] = -1
        accumulation = 0
        count = 1
        for i in range(1, self.N):
            temp_dist = np.zeros((i, 1))
            for j in range(i):
                temp_dist[j] = dist[ordrho[i], ordrho[j]]
            delta[ordrho[i]] = np.min(temp_dist[:])
            id_mind = np.argmin(temp_dist[:])
            accumulation += delta[ordrho[i]]
            conn_dist = accumulation / count
            count += 1
            parent[ordrho[i]] = ordrho[id_mind]
            nchild[ordrho[id_mind]] += 1
            delta[ordrho[0]] = np.max(delta)

        gamma = np.multiply(rho, delta.reshape(self.N,))
        gamma_sorted = np.array(sorted(gamma, reverse = True))
        ord_gamma = np.lexsort((gamma_sorted, gamma))[::-1]
        thresholdgamma = np.mean(gamma_sorted) + 4.0 * np.std(gamma_sorted, ddof = 1)
        alpha = np.zeros((20, 1))
        for i in range(20):
            alpha[i] =  delta[ord_gamma[i]] /rho[ord_gamma[i]]
        return ord_gamma, parent, nchild, gamma, gamma_sorted, alpha

    def DPD_Cluster(self, n_clusters, ordrho, ord_gamma, parent):
        classes = -np.ones((self.N, 1), dtype=np.int)
        peak_root = np.zeros((n_clusters, self.dim+1))
        local_core = -np.ones((self.N, 1), dtype=np.int)
        cores = -np.ones((n_clusters, 1), dtype=np.int)
        for i in range(n_clusters):
            classes[ord_gamma[i]] = i+1
            local_core[ord_gamma[i]] = ord_gamma[i]
            peak_root[i,:self.dim] = self.A[ord_gamma[i], :]
            peak_root[i, self.dim] = ord_gamma[i]
            cores[i] = int(peak_root[i, 2])
        for i in range(self.N):
            if classes[ordrho[i]] == -1:
                classes[ordrho[i]] = classes[parent[ordrho[i]]]
                local_core[ordrho[i]] = local_core[parent[ordrho[i]]]
        # classes[np.where(self.nb==0)[0]] = -1
        return classes, peak_root

    def Leaf_node(self, peak_root, parent, nchild):
        n_clusters = peak_root.shape[0]
        childindex = np.where(nchild==0)[0]
        for i in range(n_clusters):
            if nchild[parent[int(peak_root[i,self.dim])]] == 1:
                childindex = np.append(childindex, parent[int(peak_root[i,self.dim])])
        return childindex

    def ComputeVI(self, peak_root, classes, childindex, shortpath, rho):
        # mins, maxs = np.min(shortpath), np.max(shortpath)
        # shortpath = (shortpath-mins)/(maxs-mins)
        # mins, maxs = np.min(self.dist), np.max(self.dist)
        # dist = (self.dist-mins)/(maxs-mins)
        # minr, maxr = np.min(rho), np.max(rho)
        # rho = (rho-minr)/(maxr-minr)
        dist = self.dist
        n_clusters = peak_root.shape[0]
        s=set()
        s.update(childindex)
        intra = np.zeros((n_clusters, 1))
        inter = np.zeros((n_clusters, 1))
        inter_den = np.zeros((n_clusters, 1))
        peak_dis = np.zeros((n_clusters, 1))
        IV = np.zeros((n_clusters, 1))
        IV1 = np.zeros((n_clusters, 1))
        IV2 = np.zeros((n_clusters, 1))
        inter_M = np.zeros((n_clusters, n_clusters))
        inter_den_M = np.zeros((n_clusters, n_clusters))
        peak_dis_M = np.zeros((n_clusters, n_clusters))

        seg = 0
        for i in range(n_clusters):
            ind = np.where(classes==i+1)[0]
            if len(ind)==1: intra[i] = 0
            else:
                sc = set()
                sc.update(ind)
                sc = list(sc & s)
                n = peak_root[i, self.dim]
                intra[i] = np.mean(shortpath[int(n), sc])
            
            for j in range(i+1, n_clusters):
                ind_j = np.where(classes==j+1)[0]
                ind_i = np.where(classes==i+1)[0]
                num_n = len(ind_i)+len(ind_j)
                dis = []
                den = []
                for k in range(3):
                    if len(ind_i)==0 or len(ind_j)==0: break
                    # graphdis = shortpath[ind_i, :][:,ind_j]
                    graphdis = dist[ind_i, :][:,ind_j]
                    mindis = np.min(graphdis)
                    dis.append(mindis)
                    ind_k = np.argwhere(graphdis == mindis)[0]
                    ind_i = np.delete(ind_i, ind_k[0])
                    ind_j = np.delete(ind_j, ind_k[1])
                    den.append(rho[ind_k[0]])
                    den.append(rho[ind_k[1]])
                inter_den_M[i,j] = np.mean(den)
                inter_den_M[j,i] = inter_den_M[i,j]
                inter_M[i, j] = np.mean(dis)#-np.std(dis)
                inter_M[j, i] = inter_M[i, j]
                # peak_dis_M[i,j] = shortpath[int(peak_root[i,self.dim]), int(peak_root[j,self.dim])] + self.dist[int(peak_root[i,self.dim]), int(peak_root[j,self.dim])]
                peak_dis_M[i,j] = shortpath[int(peak_root[i,self.dim]), int(peak_root[j,self.dim])]
                # peak_dis_M[i,j] = self.dist[int(peak_root[i,self.dim]), int(peak_root[j,self.dim])]
                peak_dis_M[j,i] = peak_dis_M[i, j]
                seg += peak_dis_M[i,j]*inter_M[i,j]/inter_den_M[i,j]*num_n
            tmp = inter_M[i, :]
            tmp1 = np.delete(tmp, i) #delete zero
            inter[i] = np.min(tmp1)
            ind_peak = np.where(tmp==inter[i])[0][0]
            inter_den[i] = inter_den_M[i, ind_peak]
            peak_dis[i] = peak_dis_M[i, ind_peak]
            if intra[i] == 0: 
                IV[i] = 0
                IV1[i] = 0
                IV2[i] = 0
            else: 
                # IV[i] = inter[i] * len(ind)
                IV1[i] = peak_dis[i] / (inter_den[i]/inter[i]) * len(ind)
                # IV[i] = rho[int(peak_root[i,self.dim])]/intra[i] * len(ind)*inter[i]/peak_dis[i]
                # IV[i] = (inter[i]-intra[i]) / max(inter[i], intra[i]) * len(ind)
                # IV[i] = (peak_dis[i]-intra[i]) / max(peak_dis[i], intra[i]) * len(ind)*inter[i]
                # IV[i] = rho[int(peak_root[i,self.dim])]/intra[i] * peak_dis[i] /(inter_den[i]/inter[i]) * len(ind)
                # IV[i] = np.sum(rho[ind])/intra[i] * peak_dis[i] /(inter_den[i]/inter[i]) * len(ind)**2
                IV2[i] = np.sum(rho[ind])/intra[i]* len(ind)
                IV[i] = np.sum(rho[ind])/intra[i] /(inter_den[i]/inter[i]) * len(ind)**2
                # IV2[i] = 1 / (inter_den[i]/inter[i]) * len(ind)
                # print('   ',np.sum(rho[ind]), intra[i], len(ind), np.sum(rho[ind])/intra[i]/len(ind))
        # print('peak dis',peak_dis.reshape(n_clusters,))
        # print('dis',inter.reshape(n_clusters,))
        # print('density',inter_den.reshape(n_clusters,))
        # print(intra)
        # return np.sum(IV2)*seg / self.N**2/n_clusters/(n_clusters-1), np.sum(IV1) / self.N, np.sum(IV2)/self.N
        return np.sum(IV2)*np.sum(IV1) / self.N**2, np.sum(IV1) / self.N, np.sum(IV2)/self.N
