#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import division
import numpy as np
import sys
import math
import matplotlib.pyplot as plt
import pickle


def main(file_path):
    """
    This function runs your code for problem 2.

    You can also use this to test your code for problem 1,
    but make sure that you do not leave anything in here that will interfere
    with problem 2. Especially make sure that gmm_est does not output anything
    extraneous, as problem 2 has a very specific expected output.
    """
    # YOUR CODE FOR PROBLEM 2 GOES HERE
    X_train, X_train2 = read_gmm_file(file_path)
    log_possi = []
    mu_, sigmasq_, wt_, L_ = 0, 0, 0, 0
    for its in range(20):
        mu_init = np.array([min(X_train), max(X_train)])
        sigma_init = np.array([1.0, 1.0])
        wt_init = np.array([0.5, 0.5])
        its = its + 1
        mu_, sigmasq_, wt_, L_ = gmm_est(X_train, mu_init, sigma_init, wt_init, its)
        log_possi.append(L_)
    plt.xlabel('times of iteration')
    plt.ylabel('log possibility')
    plt.title('log possibility of 20 times of iteration')
    i = list(np.array(range(len(log_possi))) + 1)
    plt.plot(i, log_possi, 'k')
    plt.savefig('plotclass1.png')
    plt.close('all')
    log_possi2 = []
    mu2_, sigmasq2_, wt2_, L2_ = 0, 0, 0, 0
    for its in range(20):
        mu_init2 = np.array([-10.0, -1.0, 20.0])
        sigma_init2 = np.array([3.0, 3.0, 5.0])
        wt_init2 = np.array([0.25, 0.25, 0.5])
        its = its + 1
        mu2_, sigmasq2_, wt2_, L2_ = gmm_est(X_train2, mu_init2, sigma_init2, wt_init2, its)
        log_possi2.append(L2_)
    plt.xlabel('times of iteration')
    plt.ylabel('log possibility')
    plt.title('log possibility of 20 times of iteration')
    i = list(np.array(range(len(log_possi2))) + 1)
    plt.plot(i, log_possi2, 'k')
    plt.savefig('plotclass2.png')
    plt.close('all')



    # mu_results1, sigma2_results1, w_results1 are all numpy arrays
    # with learned parameters from Class 1
    print 'Class 1'
    print 'mu =', mu_, '\nsigma^2 =', sigmasq_, '\nw =', wt_

    # mu_results2, sigma2_results2, w_results2 are all numpy arrays
    # with learned parameters from Class 2
    print '\nClass 2'
    print 'mu =', mu2_, '\nsigma^2 =', sigmasq2_, '\nw =', wt2_
    output = [[mu_, sigmasq_, wt_], [mu2_, sigmasq2_, wt2_]]
    f3 = open("parameter.txt", 'wb')
    pickle.dump(output, f3)
    f3.close()
    return [mu_, sigmasq_, wt_], [mu2_, sigmasq2_, wt2_]


def responsibility(mu, sigma, wt, x, j):
    pi = math.pi
    x_ = []

    for i in range(len(mu)):
        x_.append(x)
    x_ = np.array(x_)
    N = 1.0/((pi*2)**0.5*sigma**0.5) * ((math.e)**((-(x_-mu)**2)/(2*sigma)))
    #N = -(x_-mu)**2/(2*sigma)-np.log(sigma)
    N = np.array(N)
    res = wt[j]*N[j] / sum(wt*N)
    return res


def gmm_est(X, mu_init, sigmasq_init, wt_init, its):
    """
    Input Parameters:
      - X             : N 1-dimensional data points (a 1-by-N numpy array)
      - mu_init       : initial means of K Gaussian components (a 1-by-K numpy array)
      - sigmasq_init  : initial  variances of K Gaussian components (a 1-by-K numpy array)
      - wt_init       : initial weights of k Gaussian components (a 1-by-K numpy array that sums to 1)
      - its           : number of iterations for the EM algorithm

    Returns:
      - mu            : means of Gaussian components (a 1-by-K numpy array)
      - sigmasq       : variances of Gaussian components (a 1-by-K numpy array)
      - wt            : weights of Gaussian components (a 1-by-K numpy array, sums to 1)
      - L             : log likelihood
    """
    respon = []
    # YOUR CODE FOR PROBLEM 1 HERE

    for i in range(its):
        for j in range(len(mu_init)):
            sum_res = 0.0
            sum_mu = 0.0
            sum_sig = 0.0
            res_array =[]
            for ele in X:
                res = responsibility(mu_init, sigmasq_init, wt_init, ele, j)
                res_array.append(res)
                sum_res += res
                sum_mu += res * ele
                #sum_sig += float(res * (ele - mu_init[j]) ** 2)
            wt_init[j] = sum_res/len(X)
            mu_init[j] = sum_mu/sum_res
            for i in range(len(X)):
                sum_sig += float(res_array[i] * (X[i] - mu_init[j]) ** 2)
            sigmasq_init[j] = sum_sig/sum_res

    mu = mu_init
    sigmasq = sigmasq_init
    wt = wt_init
    possibility = 0
    for i in range(len(X)):
        P = 0
        for j in range(len(wt)):
            P += wt[j]*(1.0/((np.pi*2)**0.5*sigmasq[j]**0.5) * (math.e)**((-(X[i]-mu[j])**2)/(2*sigmasq[j])))
        possibility += np.log(P)
    L = possibility
    #print L



    return mu, sigmasq, wt, L


def read_gmm_file(path_to_file):
    """
    Reads either gmm_test.csv or gmm_train.csv
    :param path_to_file: path to .csv file
    :return: two numpy arrays for data with label 1 (X1) and data with label 2 (X2)
    """
    X1 = []
    X2 = []

    data = open(path_to_file).readlines()[1:] # we don't need the first line
    for d in data:
        d = d.split(',')

        # We know the data is either class 1 or class 2
        if int(d[1]) == 1:
            X1.append(float(d[0]))
        else:
            X2.append(float(d[0]))

    X1 = np.array(X1)
    X2 = np.array(X2)
    # class1 = X1
    # class2 = X2
    # bins = 50  # the number 50 is just an example.
    # plt.subplot(2,1,1)
    # plt.hist(class1, bins)
    # plt.subplot(2, 1, 2)
    # plt.hist(class2, bins)
    # plt.show()
    return X1, X2

if __name__ == '__main__':
    file_path = sys.argv[1]
    main(file_path)
