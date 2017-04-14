#	Starter code for linear regression problem
#	Below are all the modules that you'll need to have working to complete this problem
#	Some helpful functions: np.polyfit, scipy.polyval, zip, np.random.shuffle, np.argmin, np.sum, plt.boxplot, plt.subplot, plt.figure, plt.title
import sys
import csv
import numpy as np
import scipy
import matplotlib.pyplot as plt
import math


def polyfit(X, Y, n):
    xMat = np.mat(X);
    yMat = np.mat(X).T
    xTx = xMat.T * xMat
    if np.linalg.det(xTx) == 0.0:
        print "this matrix is singular, cannot do inverse"
        return
    omega = xTx.I * (xMat.T * yMat)
    return omega


def computBasis(X, k, n):  # n is the number of instances, k is max order
    phi_X = [[0 for col in range(k)] for row in range(n)]  # an n*k matrix
    for row in phi_X:
        for col in row:
            phi_X[row][col] = X[row] ** col
    return phi_X


def computMse(X, Y, omega):
    mse = 0
    xMat = np.mat(X);
    yMat = np.mat(Y);
    omegaMat = np.mat(omega)
    for i in range(len(X)):
        mse += (omegaMat.T * xMat - yMat).T * (omegaMat.T * xMat - yMat)
    mse = mse / float(len(Y))
    return mse


def nfoldpolyfit(X, Y, maxK, n, verbose):
    #	NFOLDPOLYFIT Fit polynomial of the best degree to data.
    #   NFOLDPOLYFIT(X,Y,maxDegree, nFold, verbose) finds and returns the coefficients
    #   of a polynomial P(X) of a degree between 1 and N that fits the data Y
    #   best in a least-squares sense, averaged over nFold trials of cross validation.
    #
    #   P is a vector (in numpy) of length N+1 containing the polynomial coefficients in
    #   descending powers, P(1)*X^N + P(2)*X^(N-1) +...+ P(N)*X + P(N+1). use
    #   numpy.polyval(P,Z) for some vector of input Z to see the output.
    #
    #   X and Y are vectors of datapoints specifying  input (X) and output (Y)
    #   of the function to be learned. Class support for inputs X,Y:
    #   float, double, single
    #
    #   maxDegree is the highest degree polynomial to be tried. For example, if
    #   maxDegree = 3, then polynomials of degree 0, 1, 2, 3 would be tried.
    #
    #   nFold sets the number of folds in nfold cross validation when finding
    #   the best polynomial. Data is split into n parts and the polynomial is run n
    #   times for each degree: testing on 1/n data points and training on the
    #   rest.
    #
    #   verbose, if set to 1 shows mean squared error as a function of the
    #   degrees of the polynomial on one plot, and displays the fit of the best
    #   polynomial to the data in a second plot.
    #
    #
    #   AUTHOR: Xinzuo Wang (This is where you put your name)

    #   For problem 1, you can use numpy.polyfit and numpy.polyval funciton.

    testSize = int(float(len(X)) / float(n))
    bestK = -1;
    bestTestE = float("inf");
    mse_k = []
    for k in range(maxK):
        meanTestE = 0
        for fold in range(n):
            testX = X[fold * testSize: (fold + 1) * testSize]
            trainX = [item for item in X if item not in testX]

            testY = Y[fold * testSize: (fold + 1) * testSize]
            trainY = [item for item in Y if item not in testY]

            omega = np.polyfit(trainX, trainY, k)
            testE = 0
            for i, item in enumerate(testY):
                testE += (np.polyval(omega, testX[i]) - testY[i]) ** 2
            testE = math.sqrt(testE)
            meanTestE += testE
        meanTestE = meanTestE / float(n)
        mse_k.append(meanTestE)
        if bestTestE > meanTestE:
            bestTestE = meanTestE;
            bestK = k
    print "bestK = ", bestK
    best_omega = np.polyfit(X, Y, bestK)
    Z = np.linspace(min(X), max(X), 100)
    plt.figure(1)
    plt.plot(range(0, maxK), mse_k);

    plt.figure(2)
    plt.plot(X, Y, '.');
    plt.plot(Z, np.polyval(best_omega, Z))
    plt.show()

    print "The predict value of x = 3 is : ", np.polyval(best_omega, 3)
    return 0


def main():
    # read in system arguments, first the csv file, max degree fit, number of folds, verbose
    rfile = sys.argv[1]
    maxK = int(sys.argv[2])
    nFolds = int(sys.argv[3])
    verbose = bool((sys.argv[4]))
    csvfile = open(rfile, 'rb')

    #    rfile = 'linearreg.csv'
    #    maxK = 10
    #    nFolds = 10
    #    print "Thre choice of value n is:", nFolds
    #    verbose = False

    csvfile = open(rfile, 'rb')
    dat = csv.reader(csvfile, delimiter=',')
    X = []
    Y = []
    # put the x coordinates in the list X, the y coordinates in the list Y
    for i, row in enumerate(dat):
        if i > 0:
            X.append(float(row[0]))
            Y.append(float(row[1]))
    X = np.array(X)
    Y = np.array(Y)

    nfoldpolyfit(X, Y, maxK, nFolds, verbose)


if __name__ == "__main__":
    main()
