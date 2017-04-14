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
        if i > 0:
            userid.append(int(row[0]))
            movieid.append(int(row[1]))
            rating.append(int(row[2]))
            timestamp.append(int(row[3]))

    moviename = []
    j = 0
    for element in movieid:

        if dict.get(element, None) == None:
            helper = []
            moviename.append(element)
            helper.append([userid[j], rating[j]])
            dict[element] = helper[:]
        else:

            helper = []
            helper = dict.get(element, None)
            helper.append([userid[j], rating[j]])
            dict[element] = helper[:]
        j += 1

    namesort = []
    for things in userid:

        if things in namesort:
            pass
        else:
            namesort.append(things)

    dict2 = dict.copy()
    for num in dict2:
        dict2[num] = []
    time = 0
    for ele in namesort:
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

    f = open('dump2.txt', 'wb')
    pickle.dump(dict2, f)
    f.close()

    f2 = open("position2.txt", 'wb')
    pickle.dump(namesort, f2)
    f2.close()





def correlation(array1, array2):
    a, b = scipy.stats.pearsonr(array1, array2)
    return 1-a


def manhattan(array1, array2):
    array1 = np.array(array1)
    array2 = np.array(array2)
    distance = sum(abs(array1 - array2))
    return distance


def item_based_cf(userid, movieid, distance, k, iFlag, position_user_sorce, dict):



    position_user = position_user_sorce.index(userid)

    distance_array = []
    distance_id = []
    array1 = dict[movieid]
    if distance == 1:
        for i in dict:
            if i != movieid:
                array2 = dict[i]
                distance_array.append(manhattan(array1, array2))
                distance_id.append(i)
    elif distance == 0:
        for i in dict:
            if i != movieid:
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
            value = dict[distance_id[position]][position_user]
            if value == 0:
                del distance_array[position]
                del distance_id[position]
            else:
                name.append(distance_id[position])
                del distance_array[position]
                del distance_id[position]
                i -= 1

    for element in name:
        rank.append(dict[element][position_user])
    bestfre = 0
    bestrank = 0
    for things in rank:
        fre = rank.count(things)
        if fre > bestfre:
            bestfre = fre
            bestrank = things

    predictedRating = bestrank
    trueRating = dict[userid][position_user]

    return trueRating, predictedRating



def problem4(datafile, distance, k, iFlag):
    f = open('namesample.txt', 'rb')
    sample = pickle.load(f)
    f.close()
    print 'namesample loaded'
    # dict, position_movie = vect_generate(datafile, movieid)
    f1 = open('dump2.txt', 'rb')
    dict = pickle.load(f1)
    f1.close()
    print 'dictionary loaded'
    dictcopy = dict.copy()
    f2 = open('position2.txt', 'rb')
    position_user_sorce = pickle.load(f2)
    f2.close()
    print 'vector position loaded'
    ## here for loop is used to clear the 100 sampled data in the dictionary to make the 99900 dataset as prior data
    ## find out the position of userid and movieid and set the value to 0
    sum = 0
    num = 0
    f3 = open('50sampledislast2.txt', 'wb')
    for element in sample:

        num += 1
        print str(num) + "loop"
        dictcopy = dict.copy()
        usertest = [v[0] for v in element]
        movietest = [v[1] for v in element]
        ratingtest = [v[2] for v in element]
        for i in range(len(usertest)):
            dictcopy[movietest[i]][position_user_sorce.index(usertest[i])] = 0
        print "resort dictionary completed"

        sum = 0
        for j in range(len(usertest)):
            true, predict = item_based_cf(usertest[j], movietest[j], distance, k, iFlag,
                                          position_user_sorce, dictcopy)
            true = ratingtest[j]
            error = (abs(true - predict))**2
            sum = error + sum
            print str(usertest[j]) + '   '+str(movietest[j])
            print 'true: ' + str(true) +' predict: ' + str(predict)
        ave = float(sum)/len(usertest)
        f3.write(str(ave)+'\n')


    ave_error = float(sum) /  len(sample[0])

    print "the average error is "+ str(ave_error)








if __name__ == "__main__":
    #vect_generate('ml-100k/u.data')
    stime = time.time()
    datafile = 'ml-100k/u.data'
    distance = 0
    k = 32
    iFlag = 0
    problem4(datafile, distance, k, iFlag)
    #print user_based_cf('ml-100k/u.data', 30, 82, 0, 5, 0, 0, 0)
