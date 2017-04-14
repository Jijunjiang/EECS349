import sys
import csv
import numpy as np
import scipy

def check(w_init, X, Y):
    multi = 0
# check if every data in source is classified correctly
    for i in range(len(X)):
        multi = (w_init[1] * X[i] + w_init[0] + w_init[2]*X[i]*X[i]) * Y[i]
        if multi < 0:
            print i
            return True
        else:
            pass

    return False




def perceptrona(w_init, X, Y):
    # figure out (w, k) and return them here. w is the vector of weights, k is how many iterations it took to converge.
    k = 0
    while check(w_init, X, Y):
        for i in range(len(X)):
            if (w_init[2]*X[i]*X[i] +w_init[1] * X[i] + w_init[0]) * Y[i] < 0:
                w_init[0] = w_init[0] + Y[i]
                w_init[1] = w_init[1] + Y[i] * X[i]
                w_init[2] = w_init[2] + Y[i] * X[i]*X[i]
                print i
                print w_init

            else:
                pass
            k += 1
    return [w_init, k]


# PERCEPTRONA imply perceptron method to do classification of data with label {0, 1}
# w_iniit is the initial value of weight
#  K is the time of iteration
#   AUTHOR: Jijun Jiang

def main():
    rfile = sys.argv[1]


    # read in csv file into np.arrays X1, X2, Y1, Y2
    csvfile = open(rfile, 'rb')
    dat = csv.reader(csvfile, delimiter=',')
    X1 = []
    Y1 = []
    X2 = []
    Y2 = []
    for i, row in enumerate(dat):
        if i > 0:
            X1.append(float(row[0]))
            X2.append(float(row[1]))
            Y1.append(float(row[2]))
            Y2.append(float(row[3]))
    X1 = np.array(X1)
    X2 = np.array(X2)
    Y1 = np.array(Y1)
    Y2 = np.array(Y2)
    w_init = np.array([0.1, 0.1, 0.1])
    [w2, k2] = perceptrona(w_init, X2, Y2)
    print 'the weight is : ' + str(w2) + '     the time of iteration is : ' + str(k2)
    return (w2, k2)


if __name__ == "__main__":
    main()
