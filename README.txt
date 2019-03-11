Self-Orgnization Maps(SOM)

1.ANN 
    two layers, competition-cooperation, spatial similarity
  learning rate: from one to zero
  the size of grid: large enough to classify
  neighborhood radius: half of grid

Algorithm£º
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