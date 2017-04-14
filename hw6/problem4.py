#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import division
import numpy as np
import sys
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import gmm_est
import math
import pickle


def main():
    """
    This function runs your code for problem 3.

    You can use this code for problem 4, but make sure you do not
    interfere with what you need to do for problem 3.
    """
    file_path = 'gmm_test.csv'#sys.argv[1]

    # YOUR CODE FOR PROBLEM 3 GOES HERE
    X1, X2, X, class_flag = read_gmm_file(file_path)
    X1_pre = []
    X2_pre = []
    #[mu1, sigmasq1, wt1],[mu2, sigmasq2, wt2] = gmm_est.main('gmm_train.csv')
    f = open('parameter.txt', 'rb')
    sample = pickle.load(f)
    f.close()
    [mu1, sigmasq1, wt1], [mu2, sigmasq2, wt2] = sample[0], sample[1]
    #p1 = len(X1)/(len(X1)+len(X2))
    p1 = float(len(X1))/(len(X1)+len(X2))
    class_pred = gmm_classify(X, mu1, sigmasq1, wt1, mu2, sigmasq2, wt2, p1)
    count = 0
    for i in range(len(X)):
        if class_pred[i] == 1:
            X1_pre.append(X[i])
        else:
            X2_pre.append(X[i])
        if class_pred[i] != class_flag[i]:
            count += 1
    correct_rate = 1.0 - float(count)/len(X)
    print 'the correct rate is: ' + str(correct_rate)



    bins = 100  # the number 50 is just an example. plt.subplot(2,1,1)
    red_patch = mpatches.Patch(color='red', alpha=0.5, label='class1')

    blue_patch = mpatches.Patch(color='blue', alpha=0.5, label='class2')


    plt.legend(handles=[red_patch, blue_patch], loc=1, prop={'size': 15})
    plt.scatter(X1_pre, list(np.zeros(len(X1_pre))-1), color='red', s = 15, alpha=0.1)
    plt.hist(X1, bins, facecolor='red', alpha=0.5)
    plt.scatter(X2_pre, list(np.zeros(len(X2_pre))-1), color='blue', s = 15, alpha=0.1)
    plt.hist(X2, bins, facecolor='blue', alpha=0.5)
    plt.title('histogram and points plot for problem4')
    plt.xlabel('the value of data')
    plt.ylabel('the number of data')

    plt.show()



def gmm_classify(X, mu1, sigmasq1, wt1, mu2, sigmasq2, wt2, p1):
    """
    Input Parameters:
        - X           : N 1-dimensional data points (a 1-by-N numpy array)
        - mu1         : means of Gaussian components of the 1st class (a 1-by-K1 numpy array)
        - sigmasq1    : variances of Gaussian components of the 1st class (a 1-by-K1 numpy array)
        - wt1         : weights of Gaussian components of the 1st class (a 1-by-K1 numpy array, sums to 1)
        - mu2         : means of Gaussian components of the 2nd class (a 1-by-K2 numpy array)
        - sigmasq2    : variances of Gaussian components of the 2nd class (a 1-by-K2 numpy array)
        - wt2         : weights of Gaussian components of the 2nd class (a 1-by-K2 numpy array, sums to 1)
        - p1          : the prior probability of class 1.

    Returns:
        - class_pred  : a numpy array containing results from the gmm classifier
                        (the results array should be in the same order as the input data points)
    """

    # YOUR CODE FOR PROBLEM 3 HERE
    class_pred = []

    for ele in X:
        P1 = 0
        P2 = 0
        for j in range(len(wt1)):
            P1 += wt1[j]*(1.0/((np.pi*2)**0.5*sigmasq1[j]**0.5) * (math.e)**((-(ele-mu1[j])**2)/(2*sigmasq1[j])))
        for j in range(len(wt2)):
            P2 += wt2[j]*(1.0/((np.pi*2)**0.5*sigmasq2[j]**0.5) * (math.e)**((-(ele-mu2[j])**2)/(2*sigmasq2[j])))
        P1 = P1*p1
        P2 = P2*(1 - p1)
        if P1 >= P2:
            class_pred.append(1)
        else:
            class_pred.append(2)

    return np.array(class_pred)


def read_gmm_file(path_to_file):
    """
    Reads either gmm_test.csv or gmm_train.csv
    :param path_to_file: path to .csv file
    :return: two numpy arrays for data with label 1 (X1) and data with label 2 (X2)
    """
    X1 = []
    X2 = []
    X = []
    class_flag = []
    data = open(path_to_file).readlines()[1:] # we don't need the first line
    for d in data:
        d = d.split(',')

        # We know the data is either class 1 or class 2
        if int(d[1]) == 1:
            X1.append(float(d[0]))
            X.append(float(d[0]))
            class_flag.append(int(d[1]))
        else:
            X2.append(float(d[0]))
            X.append(float(d[0]))
            class_flag.append(int(d[1]))
    X1 = np.array(X1)
    X2 = np.array(X2)
    X = np.array(X)
    class_flag = np.array(class_flag)

    return X1, X2, X, class_flag

if __name__ == '__main__':
    main()
