# coding=utf-8
# Starter code for uesr-based collaborative filtering
# Complete the function user_based_cf below. Do not change it arguments and return variables.
# Do not change main() function,

# import modules you need here.
import sys
import scipy.stats
import csv
import numpy as np
import pickle
import time
import random
import copy


def vect_generate(datafile):
    userid = []
    movieid = []
    rating = []
    timestamp = []
    dict = {}
    dict2 = {}
    rfile = datafile
    readfile = open(rfile, 'rb')
    data = csv.reader(readfile, delimiter='\t')
    for i, row in enumerate(data):
        if i >= 0:
            userid.append(int(row[0]))
            movieid.append(int(row[1]))
            rating.append(int(row[2]))
            timestamp.append(int(row[3]))


    userid_copy = []
    movieid_copy = []
    rating_copy = []
    pre_data = []

    l = range(len(userid))
    name_list = []
    for k in range(50):
        line = []
        pre_line = []
        samp_list = random.sample(l, 100)
        for element in samp_list:
            line.append([userid[element], movieid[element], rating[element]])
        name_list.append(line)

    f3 = open("namesample.txt", 'wb')
    pickle.dump(name_list, f3)
    f3.close()


    usename = []
    userid_ = 0
    j = 0
    for element in userid:

        if dict.get(element, None) == None:
            helper = []
            usename.append(element)
            helper.append([movieid[j], rating[j]])
            dict[element] = helper[:]
        else:

            helper = []
            helper = dict.get(element, None)
            helper.append([movieid[j], rating[j]])
            dict[element] = helper[:]
        j += 1

    moviesort = []
    for things in movieid:

        if things in moviesort:
            pass
        else:
            moviesort.append(things)

    dict2 = dict.copy()
    for num in dict2:
        dict2[num] = []
    time = 0
    for ele in moviesort:
        for i in dict:
            for j in range(len(dict[i])):
                if ele == dict[i][j][0]:
                    dict2[i].append(int(dict[i][j][1]))
            if ele in [v[0] for v in dict[i]]:
                pass
            else:
                dict2[i].append(0)

        time += 1
        print time

    f = open('dump_problem4.txt', 'wb')
    pickle.dump(dict2, f)
    f.close()

    f2 = open("position_problem4.txt", 'wb')
    pickle.dump(moviesort, f2)
    f2.close()





def correlation(array1, array2):
    a, b = scipy.stats.pearsonr(array1, array2)
    return 1-a


def manhattan(array1, array2):
    array1 = np.array(array1)
    array2 = np.array(array2)
    distance = sum(abs(array1 - array2))
    return distance


def user_based_cf(userid, movieid, distance, k, iFlag, position_movie_sorce, dict):



    position_movie = position_movie_sorce.index(movieid)

    distance_array = []
    distance_id = []
    array1 = dict[userid]
    if distance == 1:
        for i in dict:
            if i != userid:
                array2 = dict[i]
                distance_array.append(manhattan(array1, array2))
                distance_id.append(i)
    elif distance == 0:
        for i in dict:
            if i != userid:
                array2 = dict[i]
                distance_array.append(correlation(array1, array2))
                distance_id.append(i)


    else:
        print "pls input 0 pr 1 to choose distance type!"
        return 0
    name = []
    rank = []
    if iFlag == 1:
        for i in range(k):
            position = distance_array.index(min(distance_array))
            #print min(distance_array)
            name.append(distance_id[position])
            del distance_array[position]
            del distance_id[position]
    elif iFlag == 0:
        i = k
        while i > 0:
            if distance_array == []:
                break
            position = distance_array.index(min(distance_array))

            #print max(distance_array)
            value = dict[distance_id[position]][position_movie]
            if value == 0:
                del distance_array[position]
                del distance_id[position]
            else:
                name.append(distance_id[position])
                del distance_array[position]
                del distance_id[position]
                i -= 1

    for element in name:
        rank.append(dict[element][position_movie])
    bestfre = 0
    bestrank = 0
    for things in rank:
        fre = rank.count(things)
        if fre > bestfre:
            bestfre = fre
            bestrank = things

    predictedRating = bestrank
    trueRating = dict[userid][position_movie]

    return trueRating, predictedRating



def problem4(datafile, distance, k, iFlag):
    f = open('namesample.txt', 'rb')
    sample = pickle.load(f)
    f.close()
    print 'namesample loaded'
    # dict, position_movie = vect_generate(datafile, movieid)
    f1 = open('dump.txt', 'rb')
    dict = pickle.load(f1)
    f1.close()
    print 'dictionary loaded'
    dictcopy = dict.copy()
    f2 = open('position.txt', 'rb')
    position_movie_sorce = pickle.load(f2)
    f2.close()
    print 'vector position loaded'
    ## here for loop is used to clear the 100 sampled data in the dictionary to make the 99900 dataset as prior data
    ## find out the position of userid and movieid and set the value to 0
    sum = 0
    num = 0
    f3 = open('50samplelast1.txt', 'wb')
    for element in sample:

        num += 1
        #print str(num) + "loop"
        dictcopy = dict.copy()
        usertest = [v[0] for v in element]
        movietest = [v[1] for v in element]
        ratingtest = [v[2] for v in element]
        for i in range(len(usertest)):
            dictcopy[usertest[i]][position_movie_sorce.index(movietest[i])] = 0
        #print "resort dictionary completed"

        sum = 0
        for j in range(len(usertest)):

            true, predict = user_based_cf(usertest[j], movietest[j], distance, k, iFlag,
                                          position_movie_sorce, dictcopy)
            true = ratingtest[j]
            error = (abs(true - predict))**2
            sum = error + sum
            print str(usertest[j]) + '   '+str(movietest[j])
            print 'true: ' + str(true) +' predict: ' + str(predict)
        ave = float(sum) / len(usertest)
        print '//////////////////////////////////////////    ' + str(ave)
        f3.write(str(ave) + '\n')













if __name__ == "__main__":
    #vect_generate('/Users/apple/Desktop/reading/349/eecs349-fall16-hw4-1/ml-100k/u.data')
    stime = time.time()
    #print 'k is ' + str(k) +':'
    datafile = '/Users/apple/Desktop/reading/349/eecs349-fall16-hw4-1/ml-100k/u.data'
    distance = 0
    iFlag = 0
    k = 32
    problem4(datafile, distance, k, iFlag)
    #print user_based_cf('/Users/apple/Desktop/reading/349/eecs349-fall16-hw4-1/ml-100k/u.data', 30, 82, 0, 5, 0, 0, 0)
    print time.time() - stime
