
import numpy as np
import scipy
import matplotlib.pyplot as plt
import matplotlib
import math
import platform
if platform.system() == 'Darwin':
    matplotlib.use('MacOSX')


def discretize(inputs, data, dmin, dmax, csummax, kmax):
    """


    :param inputs: inputs
    :param data: data
    :param dmin: minimum distance between centers
    :param dmax: maximum distance between centers
    :param csummax: square of covariance max total
    :param kmax: max number of clusters
    :return:
    """

    k = 4
    # Initial amount of clusters
    ii = 0
    csumi = 1e5
    # Hyper for initial csum, make sure it doesnt hit on first loop
    n, m = np.shape(inputs)
    # 90% of data, was testing this because clustering was weird on boundaries
    # ,so I clustered boundaries individually


    in10 = inputs[0:int(0.1*n), :]
    D10 =  data[0:int(0.1*n)]
    in90 = inputs[int(0.9*n):n]
    D90 = data[int(0.9*n):n]
    inputs = inputs[(int(0.1*n) - 1):(int(0.9*n)+1), :]
    data = data[(int(0.1*n) - 1):(int(0.9*n)+1)]
    in10c = []
    in90c = []
    for i in range(m):
        in10c.append(np.average(in10[:,i]))
        in90c.append(np.average(in90[:,i]))
    while 1:
        centers, label = scipy.cluster.vq.kmeans2(inputs, k) # K means cluster
        dist = []
        for i in range(k):
            for j in range(i, k):
                if i != j:
                    dist.append(np.linalg.norm(centers[i] - centers[j])) # Distance from each input to its center as a vector
                    ii += 1

        Dins = np.zeros([k, np.shape(inputs)[0], np.shape(inputs)[1]])
        Douts = np.zeros([k, np.shape(inputs)[0]])
        ii = np.zeros([k, 1])
        for j in range(np.shape(inputs)[0]):
            for i in range(np.shape(inputs)[1]):
                Dins[label[j], int(ii[label[j]]), i] = inputs[j, i] # Break up inputs and data into clusters defined
                Douts[label[j], int(ii[label[j]])] = data[j]

            ii[label[j]] += 1

        cin = []
        cout = []
        cov = []
        icov = []
        csum = 0
        for i in range(k):
            cin.append(Dins[i, 0:int(ii[i]), :]) # Dins is too long, Cins makes it into lists of inputs that are different sizes
            cout.append(Douts[i, 0:int(ii[i])]) # Clustered outputs same as above
            cov.append(np.cov(np.transpose(np.array(Dins[i, 0:int(ii[i]), :])))) # Covariance of each cluster
            csum = np.trace(cov[i]) + csum # Sum of covariances of clusters

        # dmin < np.min(dist) and dmax > np.max(dist) # Messing between distance check and covariance check
        if csum > csumi or k >= kmax:
            break
        else:
            k += 1
        csumi = csum # Reinitializes c to make sure loop repeats if covariances are shrinking

    # Final data clustering below

    Dins = np.zeros([k, np.shape(inputs)[0], np.shape(inputs)[1]])
    Douts = np.zeros([k,np.shape(inputs)[0]])
    ii = np.zeros([k, 1])
    for j in range(np.shape(inputs)[0]):
        for i in range(np.shape(inputs)[1]):
            Dins[label[j], int(ii[label[j]]), i] = inputs[j, i]
            Douts[label[j], int(ii[label[j]])] = data[j]

        ii[label[j]] += 1

    cin = []
    cout = []
    cov = []
    icov = []
    for i in range(k):
        cin.append(Dins[i,0:int(ii[i]),:])
        cout.append(Douts[i, 0:int(ii[i])])
        cov.append(np.cov(np.transpose(np.array(Dins[i,0:int(ii[i]),:]))))
        icov.append(np.linalg.inv(cov[i]))

    cin.append(in10)
    cin.append(in90)
    cout.append(D10)
    cout.append(D90)
    icov.append(np.linalg.inv(np.cov(np.transpose(in10))))
    icov.append(np.linalg.inv(np.cov(np.transpose(in90))))
    centers = np.vstack([centers,in10c])
    centers = np.vstack([centers, in90c])
    return centers, label, Dins, cin, cout, cov, icov

# Coverage is the same
def coverage3(betas, mtx, phis, normputs, data, draws, plots):
    """


       """

    m, mbets = np.shape(betas)  # Size of betas
    n, mputs = np.shape(normputs)  # Size of normalized inputs
    setnos_p = np.random.randint(m, size=(1, draws))  # Random draws  from integer distribution
    i = 1
    while i == 1:
        setnos = np.unique(setnos_p)
        if np.size(setnos) == np.size(setnos_p):
            i = 0
        else:
            setnos_p = np.append(setnos, np.random.randint(m, size=(1, draws - np.shape(setnos)[0])))
    X = np.zeros((n, mbets))
    normputs = np.asarray(normputs)
    for i in range(n):
        phind = []  # Rounded down point of input from 0-499
        for j in range(len(normputs[i])):
            phind.append(math.floor(normputs[i, j] * 498))
            # 499 changed to 498 for python indexing
        phind_logic = []
        for k in range(len(phind)):
            if phind[k] == 498:
                phind_logic.append(1)
            else:
                phind_logic.append(0)
        phind = np.subtract(phind, phind_logic)
        for j in range(1, mbets):
            phi = 1
            for k in range(mputs):
                num = mtx[j - 1, k]
                if num > 0:
                    xsm = 498 * normputs[i][k] - phind[k]
                    phi = phi * (phis[int(num) - 1][0][phind[k]] + phis[int(num) - 1][1][phind[k]] * xsm +
                                 phis[int(num) - 1][2][phind[k]] * xsm ** 2 + phis[int(num) - 1][3][
                                     phind[k]] * xsm ** 3)
            X[i, j] = phi
    X[:, 0] = np.ones((n,))
    modells = np.zeros((np.shape(data)[0], draws))
    for i in range(draws):
        modells[:, i] = np.matmul(X, betas[setnos[i], :])
    meen = np.mean(modells, 1)
    bounds = np.zeros((np.shape(data)[0], 2))
    cut = int(np.floor(draws * .025))
    for i in range(np.shape(data)[0]):
        drawset = np.sort(modells[i, :])
        bounds[i, 0] = drawset[cut]
        bounds[i, 1] = drawset[draws - cut]
    if plots:
        plt.plot(meen, 'b', linewidth=2)
        plt.plot(bounds[:, 0], 'k--')
        plt.plot(bounds[:, 1], 'k--')
        plt.plot(data, 'ro')
        plt.show()
    rmse = np.sqrt(np.mean(meen - data) ** 2)
    return meen, bounds, rmse


def localCoverage(betas, mtx, phis, centers, normputs, data, draws, plots, icov):
    """

    :param centers: center locations of cluster
    :param icov: inverse covariance matrix
    :return:


    """



    meen_all = []
    for i in range(np.shape(centers)[0]):
        meen, null, null = coverage3(betas[i], mtx[i], phis, normputs, data, draws, 0)
        meen_all.append(meen)
    # Gets results for all model for every cluster at every input


    dist = np.zeros([np.shape(centers)[0], np.size(data)])
    distV = np.zeros([np.shape(centers)[0], np.size(data), np.shape(normputs)[1]])

    # Total distance and vector distance from each input to all the centers clusters

    for i in range(np.shape(centers)[0]):
        for j in range(np.size(data)):
            dist[i,j] = np.linalg.norm(normputs[j,:] - centers[i])
        distV[i,:,:] = normputs - centers[i]
    tot_dist = []
    for j in range(np.size(data)):
        tot_dist = np.append(tot_dist, np.sum(dist[:,j]))
    meen_total = np.zeros([np.size(data), 1])
    meen_C = np.zeros([np.size(data),1])
    w = np.zeros([np.shape(centers)[0], np.size(data)])
    for j in range(np.size(data)):
        for i in range(np.shape(centers)[0]):
            """
            this is where i define the weights, was messing with different ones so thats why there is a few
            
            wimd = weight inverse melancholious??? distance
            
            wied = weight, inverse euclidian distance
            
            wikd = this was the weight in the paper
            
            """
            wimd = 1/np.power(np.sqrt(np.matmul(np.matmul(np.transpose(distV[i,j,:]), icov[i]),distV[i,j,:])),3)
            # wied = 1/np.power(dist[i,j],3)
            wikd = np.exp(-0.5*np.matmul(np.matmul(np.transpose(distV[i,j,:]), icov[i]),distV[i,j,:]))
            meen_total[j] = meen_total[j] + (wimd + wikd)*meen_all[i][j] # Weighted average
            w[i, j] = wimd + wikd
        # Calculate the prediction from the weights
        meen_C[j] = (meen_total[j])/np.sum(w[:,j])
    if plots:
        plt.plot(meen_C, 'b', linewidth=2)
        plt.plot(data, 'ro')
        plt.show()
    rmse = np.sqrt(np.mean(meen_C - data) ** 2)
    return meen_total, rmse
