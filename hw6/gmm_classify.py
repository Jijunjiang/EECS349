#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import division
import numpy as np
import sys
#import matplotlib.pyplot as plt
import gmm_est
import math
import pickle


def main():
    """
    This function runs your code for problem 3.

    You can use this code for problem 4, but make sure you do not
    interfere with what you need to do for problem 3.
    """
    file_path = sys.argv[1]
    class1_data = []
    class2_data = []
    # YOUR CODE FOR PROBLEM 3 GOES HERE
    X1, X2, X= read_gmm_file(file_path)
    f = open('parameter.txt', 'rb')
    sample = pickle.load(f)
    f.close()
    [mu1, sigmasq1, wt1],[mu2, sigmasq2, wt2] = sample[0], sample[1]
    #p1 = len(X1)/(len(X1)+len(X2))
    p1 = float(len(X1))/(len(X1)+len(X2))

    class_pred = gmm_classify(X, mu1, sigmasq1, wt1, mu2, sigmasq2, wt2, p1)
    for i in range(len(class_pred)):
        if class_pred[i] == 1:
            class1_data.append(X[i])
        else:
            class2_data.append(X[i])

    # class1_data is a numpy array containing
    # all of the data points that your gmm classifier
    # predicted will be in class 1.
    print 'Class 1'
    print class1_data
    #
    # # class2_data is a numpy array containing
    # # all of the data points that your gmm classifier
    # # predicted will be in class 2.
    print '\nClass 2'
    print class2_data


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
        P2 = P2*(1.0 - p1)
        if P1 >= P2:
            class_pred.append(1)
        else:
            class_pred.append(2)

    return class_pred


def read_gmm_file(path_to_file):
    """
    Reads either gmm_test.csv or gmm_train.csv
    :param path_to_file: path to .csv file
    :return: two numpy arrays for data with label 1 (X1) and data with label 2 (X2)
    """
    X1 = []
    X2 = []
    X = []
    data = open(path_to_file).readlines()[1:] # we don't need the first line
    for d in data:
        d = d.split(',')

        # We know the data is either class 1 or class 2
        if int(d[1]) == 1:
            X1.append(float(d[0]))
            X.append(float(d[0]))
        else:
            X2.append(float(d[0]))
            X.append(float(d[0]))

    X1 = np.array(X1)
    X2 = np.array(X2)
    X = np.array(X)

    return X1, X2, X

if __name__ == '__main__':
    main()
