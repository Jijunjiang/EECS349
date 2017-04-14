import csv
import numpy as np
import matplotlib.pyplot as plt


def problrm1_pre():
    userid = []
    movieid =[]
    rating = []
    timestamp = []
    dict = {}
    dict2 = {}
    rfile = 'ml-100k/u.data'
    readfile = open(rfile, 'rb')
    data = csv.reader(readfile, delimiter='\t')
    for i, row in enumerate(data):
        if i > 0:
            userid.append(row[0])
            movieid.append(row[1])
            rating.append(row[2])
            timestamp.append(row[3])

    helper = []
    usename = []
    userid_ = 0
    # restore everything in dictionary with userid is the key and the movies the userid previewed
    # is the elements in that key
    j = 0
    for element in userid:

        if dict.get(element, None) == None:
            helper = []
            usename.append(element)
            helper.append(movieid[j])
            dict[element] = helper[:]
        else:

            helper = []
            helper = dict.get(element, None)
            helper.append(movieid[j])
            dict[element] = helper[:]
        j += 1

    for id in usename:
        for otherid in usename:
            count = 0
            if otherid == id:
                pass
            else:
                for movie in dict[otherid]:

                     if movie in dict[id]:
                         count += 1
                     else:
                         pass
                if dict2.get(count, None) == None: # count is the key in dictionary means the number of common review
                    # and the relevant value in dictionary is the number of pairs
                    dict2[count] = 1
                else:
                    dict2[count] += 1
                print count
    myfile = open('result.txt', 'wb')
    for (com_r, num_p) in dict2.items():
        myfile.write(str(com_r) + ',' + str(num_p) + '\n')
    myfile.close()

def problem1():
    common_review = []
    num_pair = []
    rfile = 'result.txt'
    readfile = open(rfile, 'rb')
    data = csv.reader(readfile, delimiter=',')
    for i, row in enumerate(data):
        common_review.append(int(row[0]))
        num_pair.append(int(row[1])/2)
    y_pos = np.arange(len(common_review))

    plt.axis([min(common_review), max(common_review), 0, 1.2 * max(num_pair)])
    plt.bar(y_pos, num_pair, align='center', alpha=1)
    plt.ylabel('Num of user pairs')
    plt.xlabel('Num of common movies')
    plt.title('histogram of user pairs and common movies')

    plt.show()

    common_r_array = np.array(common_review)
    num_p_array = np.array(num_pair)
    ave = float(sum(common_r_array * num_p_array)) / sum(num_p_array)
    print ' mean number of movies two people have reviewed in common: \n' + str(ave)

    helper = sum(num_p_array)/2
    m = 0
    while helper > 0:
        helper = helper - num_p_array[m]
        m += 1

    print ' median number of movies two people have reviewed in common: \n' + str(common_r_array[m - 1])


def problem2():
    userid = []
    movieid = []
    rating = []
    timestamp = []
    dict = {}
    dict2 = {}
    rfile = 'ml-100k/u.data'
    readfile = open(rfile, 'rb')
    data = csv.reader(readfile, delimiter='\t')
    for i, row in enumerate(data):
        if i > 0:
            userid.append(row[0])
            movieid.append(row[1])
            rating.append(row[2])
            timestamp.append(row[3])

    helper = []
    usename = []
    userid_ = 0
    j = 0

    for element in movieid:
        if dict.get(element, None) == None:
            dict[element] = 1
        else:
            dict[element] += 1

    a = [v for v in dict.values()]
    b = [v for v in dict.keys()]
    best_movie = b[a.index(max(a))]
    print str(best_movie)+'  has the most views :' + str(max(a))
    print str(b[a.index(min(a))])+'  has the fewest views: ' + str(min(a))

    dict2 = [v for v in sorted(dict.values())]
    dict2.reverse()
    dict2 = (dict2)
    y_pos = np.arange(len(dict2))
    plt.axis([1,(len(dict2)), 0, 1.2 * max(dict2)])
    plt.plot(y_pos, dict2, "k", label = "line of number of review ")
    plt.xlabel("order of movie x")
    plt.ylabel("times of reviews y")
    plt.title("Number of review")
    plt.legend()
    plt.show()



if __name__ == "__main__":
    #problrm1_pre()
    problem1()
    problem2()
