# coding=utf-8
# Starter code for item-based collaborative filtering
# Complete the function item_based_cf below. Do not change its name, arguments and return variables. 
# Do not change main() function, 

# import modules you need here.
import sys
import scipy.stats
import csv
import numpy as np
import pickle
import time
import copy

# this function is used to generte the dictionary for the following test, the dictionary is saved as pickle files
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



def item_based_cf(datafile, userid, movieid, distance, k, iFlag, numOfUsers, numOfItems):
    '''
    build item-based collaborative filter that predicts the rating 
    of a user for a movie.
    This function returns the predicted rating and its actual rating.
    
    Parameters
    ----------
    <datafile> - a fully specified path to a file formatted like the MovieLens100K data file u.data 
    <userid> - a userId in the MovieLens100K data
    <movieid> - a movieID in the MovieLens 100K data set
    <distance> - a Boolean. If set to 0, use Pearson’s correlation as the distance measure. If 1, use Manhattan distance.
    <k> - The number of nearest neighbors to consider
    <iFlag> - A Boolean value. If set to 0 for user-based collaborative filtering, 
    only users that have actual (ie non-0) ratings for the movie are considered in your top K. 
    For item-based, use only movies that have actual ratings by the user in your top K. 
    If set to 1, simply use the top K regardless of whether the top K contain actual or filled-in ratings.

    returns
    -------
    trueRating: <userid>'s actual rating for <movieid>
    predictedRating: <userid>'s rating predicted by collaborative filter for <movieid>


    AUTHOR: Bongjun Kim (This is where you put your name)
    '''
    f = open('dump2.txt', 'rb')
    dict = pickle.load(f)
    f.close()

    f2 = open('position2.txt', 'rb')
    position_user_sorce = pickle.load(f2)
    f2.close()

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
            name.append(distance_id[position])
            del distance_array[position]
            del distance_id[position]
    elif iFlag == 0:
        i = k
        while i > 0:
            position = distance_array.index(min(distance_array))
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
    trueRating = dict[movieid][position_user]

    return trueRating, predictedRating


def main():
    datafile = sys.argv[1]
    userid = int(sys.argv[2])
    movieid = int(sys.argv[3])
    distance = int(sys.argv[4])
    k = int(sys.argv[5])
    i = int(sys.argv[6])
    numOfUsers = 943
    numOfItems = 1682
    # vect_generate(datafile)
    # this function above is used to generate the dictionary for the following test, if you would like to use
    # a new data set, pls run decomment this function
    trueRating, predictedRating = item_based_cf(datafile, userid, movieid, distance, k, i, numOfUsers, numOfItems)
    print 'userID:{} movieID:{} trueRating:{} predictedRating:{} distance:{} K:{} I:{}'\
    .format(userid, movieid, trueRating, predictedRating, distance, k, i)




if __name__ == "__main__":
    main()