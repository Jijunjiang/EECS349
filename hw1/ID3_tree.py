import csv
import sys
import math
import copy
import random

def read_file(file_path):
    reader = csv.reader(open(file_path))

    data_space = []  # record all the data set in file
    instrument = []  # used to temp record name of instruments
    container = []
    data_space_out = []

    num = 0
    for lines in reader:
        data_lines = []  # temp record data set
        for elements in lines:
            if num == 0:
                instrument = (elements.split("\t", -1))
            else:
                data_lines = elements
        num += 1
        if num > 1:
            data_space.append(data_lines.split("\t", -1))
        else:
            pass
    for j in range(len(instrument)):
        for things in data_space:
            container.append(things[j])
        data_space_out.append(container)
        container = []
    return [instrument, data_space_out]


# the insert data_set should be a sorted one with last list be the label list
def entropy_cal(data_set, num):
    label_set = []
    label_set = data_set[num]  # [...]
    num_true = label_set.count('true')
    num_f = len(label_set) - num_true
    x = float(num_true) / float(num_f + num_true)
    if x == 1 or x == 0:
        entropy_out = 1
    else:
        entropy_out = 0 - (x * math.log(x, 2) + (1 - x) * math.log(1 - x, 2))
    return entropy_out


def entropy(num_true, num_f):

    if num_true + num_f == 0:
        return 0
    else:
        x = float(num_true) / float(num_f + num_true)
        if x == 1 or x == 0:
            entropy_out = 1
        else:
            entropy_out = 0 - (x * math.log(x, 2) + (1 - x) * math.log(1 - x, 2))
        return entropy_out


# data_set should be already sorted, and this function would calculate and pick up the number of best feature
def best_instence(data_set):
    data_set_sub = []
    gain_best = 0.0
    gain_best_num = 0
    [num_tt, num_tf, num_ff, num_ft] = [0, 0, 0, 0]
    entropy_up = entropy_cal(data_set, -1)
    lable_set = data_set[-1]
    data_set_sub = copy.deepcopy(data_set)
    del data_set_sub[-1]
    i = 0
    for elements in data_set_sub:
        num = 0
        for things in elements:
            if things == "true":
                if lable_set[num] == "true":
                    num_tt += 1
                else:
                    num_tf += 1
            else:
                if lable_set[num] == "true":
                    num_ft += 1
                else:
                    num_ff += 1
            num += 1

        entropy_true = entropy(num_tt, num_tf)
        entropy_false = entropy(num_ft, num_ff)
        s = num_tt + num_tf + num_ft + num_ff

        gain_now = entropy_up - float(num_tt + num_tf) / float(s) * entropy_true - \
                   float(num_ft + num_ff) / float(s) * entropy_false
        if gain_best < gain_now:
            gain_best = gain_now
            gain_best_num = i
        i += 1
    return gain_best_num


def data_swift(data_set, instance, best_num):

    data_true = []
    data_false = []

    choise_tru_num = []
    choise_fal_num = []
    data_set_sub = copy.deepcopy(data_set)
    for num in range(len(data_set_sub[0])):
        if data_set_sub[best_num][num] == "true":
            choise_tru_num.append(num)
        else:
            choise_fal_num.append(num)
    instance_del = instance[best_num]
    del data_set_sub[best_num]
    del instance[best_num]

    for i in range(len(data_set_sub)):
        data_true_sub = []
        data_false_sub = []
        for ele in choise_tru_num:
            data_true_sub.append(data_set_sub[i][ele])

        for things in choise_fal_num:
            data_false_sub.append(data_set_sub[i][things])

        data_true.append(data_true_sub)
        data_false.append(data_false_sub)

    instance_ = copy.deepcopy(instance)
    return [data_true, data_false, instance_del, data_set_sub, instance_]


def majoritycount(data_set):
    num_true = data_set[-1].count("true")
    num_false = data_set[-1].count("false")
    if num_true > num_false:
        return "true"
    else:
        return "false"


def createTree(data_set, num_feature, feature_del, instance, tree_):

    if data_set[-1].count(data_set[-1][0]) == len(data_set[-1]):
        return data_set[-1][0]
    if 0 < len(data_set) < 2:
        return majoritycount(data_set)

    num_del = best_instence(data_set)

    [data_true, data_false, feature_del, data_set_new, instance_new] = data_swift(data_set, instance, num_del)

    Tree = {feature_del: {}}
    tree_.append(Tree)
    if len(data_true) > 1 and data_true.count([]) == 0:
        Tree[feature_del]['True'] = createTree(data_true, num_del, feature_del, instance_new, tree_)

    if len(data_false) > 1 and data_false.count([]) == 0:
        Tree[feature_del]['False'] = createTree(data_false, num_del, feature_del, instance_new, tree_)

    return Tree


def train_data_set(train_size, data_set):
    data = copy.deepcopy(data_set)
    data2 = copy.deepcopy(data_set)
    data_set_ = []
    data_set_train = []
    data_set_test = []
    num_choise = []
    while len(num_choise) < train_size + 1:
        for elements in range(len(data[0])):
            num = random.randint(0, len(data[0]) - 1)
            num_choise.append(num)
            num_choise = list(set(num_choise))

        for i in range(len(data_set)):
            for j in num_choise:
                data_set_.append(data[i][j])
                data2[i][j] = 'delete'
            data_set_train.append(data_set_)
            data_set_ = []

    data_set_test = [i for i in data2 if i != 'delete']

    return [data_set_train, data_set_test]


def probab_pre(data_set):
    num_true = 0
    num_false = 0
    for i in data_set[-1]:
        if i == "true":
            num_true += 1
        else:
            num_false += 1
    return num_true/(num_true + num_false)


def classifier(data_set, feature, tree):
    tree_str = tree.keys()[0]
    tree_sub = tree[tree_str]

    clas_out = 'true'
    for elements in tree_sub.keys():
        num = feature.index(elements)
        if str(data_set[num]) == str(elements):
            if type(tree_sub[elements]).__name__ == "dict":
                clas_out = classifier(data_set, feature, tree_sub[elements])
            else:
                clas_out = tree_sub[elements]

    return clas_out





def main():
    if len(sys.argv) != 5:
        print "please enter the following information in the order of:"
        print "python decisiontree.py <inputFileName> <trainingSetSize> <numberOfTrials> <verbose>"
        sys.exit()
    else:
        inputfilename = sys.argv[1]
        # it is the string of file path example: "/Users/apple/Desktop/reading/349/eecs349-fall16-hw1/IvyLeague.txt"
        tranningsetsize = sys.argv[2]
        numoftrail = sys.argv[3]
        verbose = sys.argv[4]

    [feature, data_space_out] = read_file(inputfilename)
    [data_set_traning, data_set_test] = train_data_set(tranningsetsize, data_space_out)
    prob = probab_pre(data_set_traning)
    tree_ = []
    Tree = createTree(data_set_traning, 0, 'start', feature, tree_)
    print Tree




if __name__ == '__main__':
  main()
