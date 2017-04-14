# Starter code for spam filter assignment in EECS349 Machine Learning
# Author: Prem Seetharaman (replace your name here)

import sys
import numpy as np
import os
import shutil
import math
import random




def parse(text_file):
     # This function parses the text_file passed into it into a set of words. Right now it just splits up the file by blank spaces, and returns the set of unique strings used in the file.
     content = text_file.read()
     for punctuations in '~!@#$%^&*()_+=-`||\\[]\;,./{}|:<>?""1234567890\n\t ':
         content = "/".join(content.split(punctuations))  # string
     back_up = np.unique(content.split('/'))
     back_up = list(back_up)
     del back_up[0]
     back_up = np.array(back_up)
     return set(back_up)


def writedictionary(dictionary, dictionary_filename):
    # Don't edit this function. It writes the dictionary to an output file.
    output = open(dictionary_filename, 'w')
    header = 'word\tP[word|spam]\tP[word|ham]\n'
    output.write(header)
    for k in dictionary:
        line = '{0}\t{1}\t{2}\n'.format(k, str(dictionary[k]['spam']), str(dictionary[k]['ham']))
        output.write(line)


def makedictionary(spam_directory, ham_directory, dictionary_filename):
    # Making the dictionary.

    ham = [f for f in os.listdir(ham_directory) if os.path.isfile(os.path.join(ham_directory, f))]
    spam = [f for f in os.listdir(spam_directory) if os.path.isfile(os.path.join(spam_directory, f))]

    spam_prior_probability = len(spam) / float((len(spam) + len(ham)))

    words = {}

    # These for loops walk through the files and construct the dictionary. The dictionary, words, is constructed so that words[word]['spam'] gives the probability of observing that word, given we have a spam document P(word|spam), and words[word]['ham'] gives the probability of observing that word, given a hamd document P(word|ham). Right now, all it does is initialize both probabilities to 0. TODO: add code that puts in your estimates for P(word|spam) and P(word|ham).
    for s in spam:
        if s == '.DS_Store':
            pass
        else:
            for word in parse(open(spam_directory + s)):
                if word not in words:
                    words[word] = {'spam': float(1.0 + 1.0)/len(spam), 'ham': float(1.0)/len(ham)}
                else:
                    words[word]['spam'] += float(1.0)/len(spam)
    for h in ham:
        if h == '.DS_Store':
            pass
        else:
            for word in parse(open(ham_directory + h)):
                if word not in words:
                    words[word] = {'spam': float(1.0)/len(spam), 'ham': float(1.0 + 1.0)/len(ham)}
                else:
                    words[word]['ham'] += float(1.0)/len(ham)

    # Write it to a dictionary output file.
    # writedictionary(words, dictionary_filename)

    return words, spam_prior_probability


def is_spam(content, dictionary, spam_prior_probability):
    # TODO: Update this function. Right now, all it does is checks whether the spam_prior_probability is more than half the data. If it is, it says spam for everything. Else, it says ham for everything. You need to update it to make it use the dictionary and the content of the mail. Here is where your naive Bayes classifier goes.
    Map1 = math.log(spam_prior_probability)
    Map2 = math.log(1 - spam_prior_probability)
    # words_sub = [v for v in dictionary.keys()]
    for ele in content:
        if ele in dictionary:
            #if dictionary[ele]['spam'] != 0 and dictionary[ele]['ham'] != 0:
            Map1 = Map1 + math.log(dictionary[ele]['spam'])
            Map2 = Map2 + math.log(dictionary[ele]['ham'])
            # elif dictionary[ele]['spam'] == 0:
            #     Map1 = Map1 -1000
            # elif dictionary[ele]['ham'] == 0:
            #     Map2 = Map2 - 1000

    #print 'spam:' + str(Map1) + '   ' + 'ham:' + str(Map2)

    if Map1 > Map2:
        return True
    else:
        return False
def is_spam_pri(content, dictionary, spam_prior_probability):
    if spam_prior_probability >= .5:
        return True
    else:
        return False

def spamsort(mail_directory, spam_directory, ham_directory, dictionary, spam_prior_probability):
    mail = [f for f in os.listdir(mail_directory) if os.path.isfile(os.path.join(mail_directory, f))]
    for m in mail:
        if m == '.DS_Store':
            pass
        else:
            content = parse(open(mail_directory + m))
            spam = is_spam_pri(content, dictionary, spam_prior_probability)
            if spam:
                shutil.copy(mail_directory + m, spam_directory)
            else:
                shutil.copy(mail_directory + m, ham_directory)

def main():
    # Here you can test your functions. Pass it a training_spam_directory, a training_ham_directory, and a mail_directory that is filled with unsorted mail on the command line. It will create two directories in the directory where this file exists: sorted_spam, and sorted_ham. The files will show up  in this directories according to the algorithm you developed.
    training_spam_directory = 'spam/'  # sys.argv[1]
    training_ham_directory = 'easy_ham/'  # sys.argv[2]

    test_mail_directory = 'result/'  # sys.argv[3]
    test_spam_directory = 'sorted_spam'
    test_ham_directory = 'sorted_ham'

    if not os.path.exists(test_spam_directory):
        os.mkdir(test_spam_directory)
    if not os.path.exists(test_ham_directory):
        os.mkdir(test_ham_directory)

    dictionary_filename = "dictionary.dict"

    # create the dictionary to be used
    dictionary, spam_prior_probability = makedictionary(training_spam_directory, training_ham_directory,
                                                        dictionary_filename)
    # sort the mail
    spamsort(test_mail_directory, test_spam_directory, test_ham_directory, dictionary, spam_prior_probability)

def cross_vali():
    spam_path = 'spam/'
    ham_path = 'easy_ham/'
    mail_s = [f for f in os.listdir(spam_path) if os.path.isfile(os.path.join(spam_path, f))]
    mail_h = [f for f in os.listdir(ham_path) if os.path.isfile(os.path.join(ham_path, f))]
    length_s = len(mail_s)/10
    length_h = len(mail_h)/10
    random.shuffle(mail_s)
    random.shuffle(mail_h)
    for i in range(10):
        cross = list(np.array(range(length_s)) + length_s * i)
        j = 0
        for mail in mail_s:
            if mail == '.DS_Store':
                pass
            else:
                if j in cross:
                    if not os.path.exists('test/test'+str(i)):
                        os.mkdir('test/test'+str(i))
                    shutil.copy(spam_path + mail, 'test/test'+str(i))
                else:
                    if not os.path.exists('test/spam_train'+str(i)):
                        os.mkdir('test/spam_train'+str(i))
                    shutil.copy(spam_path + mail, 'test/spam_train'+str(i))
            j += 1

    for i in range(10):
        cross_ = list(np.array(range(length_h)) + length_h * i)
        j = 0
        for mail in mail_h:
            if mail == '.DS_Store':
                pass
            else:
                if j in cross_:
                    if not os.path.exists('test/test'+str(i)):
                        os.mkdir('test/test'+str(i))
                    shutil.copy(ham_path + mail, 'test/test'+str(i))
                else:
                    if not os.path.exists('test/ham_train'+str(i)):
                        os.mkdir('test/ham_train'+str(i))
                    shutil.copy(ham_path + mail, 'test/ham_train'+str(i))
            j += 1
    print '10 cross validation completed'
    ham = [f for f in os.listdir(ham_path) if os.path.isfile(os.path.join(ham_path, f))]
    spam = [f for f in os.listdir(spam_path) if os.path.isfile(os.path.join(spam_path, f))]


    sum = 0
    for i in range(10):
        count = 0
        train_s_d = 'test/spam_train' + str(i) + '/'
        train_h_d = 'test/ham_train' + str(i) + '/'
        test_d = 'test/test' + str(i) + '/'
        print 'testing' + str(i) +'  ...'
        testing(train_s_d, train_h_d, test_d)

        ham_test = [f for f in os.listdir('sorted_ham_problem4') if os.path.isfile(os.path.join('sorted_ham_problem4', f))]
        spam_test = [f for f in os.listdir('sorted_spam_problem4') if os.path.isfile(os.path.join('sorted_spam_problem4', f))]
        for ele in ham_test:
            if ele in ham:
                count += 1
        for ele in spam_test:
            if ele in spam:
                count += 1
        correctrate = float(count)/(len(ham_test)+len(spam_test))
        print 'the correct rate is: ' + str(correctrate)
        sum += correctrate
        shutil.rmtree('sorted_ham_problem4')
        os.mkdir('sorted_ham_problem4')
        shutil.rmtree('sorted_spam_problem4')
        os.mkdir('sorted_spam_problem4')

    print 'the totel average correct rate is: ' + str(float(sum)/10)
    return float(sum)/10



def cross_pri():
    spam_path = 'spam/'
    ham_path = 'easy_ham/'
    mail_s = [f for f in os.listdir(spam_path) if os.path.isfile(os.path.join(spam_path, f))]
    mail_h = [f for f in os.listdir(ham_path) if os.path.isfile(os.path.join(ham_path, f))]
    length_s = len(mail_s) / 10
    length_h = len(mail_h) / 10
    random.shuffle(mail_s)
    random.shuffle(mail_h)
    for i in range(10):
        cross = list(np.array(range(length_s)) + length_s * i)
        j = 0
        for mail in mail_s:
            if mail == '.DS_Store':
                pass
            else:
                if j in cross:
                    if not os.path.exists('test/test' + str(i)):
                        os.mkdir('test/test' + str(i))
                    shutil.copy(spam_path + mail, 'test/test' + str(i))
                else:
                    if not os.path.exists('test/spam_train' + str(i)):
                        os.mkdir('test/spam_train' + str(i))
                    shutil.copy(spam_path + mail, 'test/spam_train' + str(i))
            j += 1

    for i in range(10):
        cross_ = list(np.array(range(length_h)) + length_h * i)
        j = 0
        for mail in mail_h:
            if mail == '.DS_Store':
                pass
            else:
                if j in cross_:
                    if not os.path.exists('test/test' + str(i)):
                        os.mkdir('test/test' + str(i))
                    shutil.copy(ham_path + mail, 'test/test' + str(i))
                else:
                    if not os.path.exists('test/ham_train' + str(i)):
                        os.mkdir('test/ham_train' + str(i))
                    shutil.copy(ham_path + mail, 'test/ham_train' + str(i))
            j += 1
    print '10 cross validation completed'
    ham = [f for f in os.listdir(ham_path) if os.path.isfile(os.path.join(ham_path, f))]
    spam = [f for f in os.listdir(spam_path) if os.path.isfile(os.path.join(spam_path, f))]

    sum = 0
    for i in range(10):
        count = 0
        train_s_d = 'test/spam_train' + str(i) + '/'
        train_h_d = 'test/ham_train' + str(i) + '/'
        test_d = 'test/test' + str(i) + '/'
        print 'testing' + str(i) + '  ...'
        testing(train_s_d, train_h_d, test_d)

        ham_test = [f for f in os.listdir('sorted_ham_problem4') if
                    os.path.isfile(os.path.join('sorted_ham_problem4', f))]
        spam_test = [f for f in os.listdir('sorted_spam_problem4') if
                     os.path.isfile(os.path.join('sorted_spam_problem4', f))]
        for ele in ham_test:
            if ele in ham:
                count += 1
        for ele in spam_test:
            if ele in spam:
                count += 1
        correctrate = float(count) / (len(ham_test) + len(spam_test))
        print 'the correct rate is: ' + str(correctrate)
        sum += correctrate
        shutil.rmtree('sorted_ham_problem4')
        os.mkdir('sorted_ham_problem4')
        shutil.rmtree('sorted_spam_problem4')
        os.mkdir('sorted_spam_problem4')

    print 'the totel average correct rate is: ' + str(float(sum) / 10)
    return float(sum) / 10


def testing(training_spam_directory, training_ham_directory, test_mail_directory):
    # Here you can test your functions. Pass it a training_spam_directory, a training_ham_directory, and a mail_directory that is filled with unsorted mail on the command line. It will create two directories in the directory where this file exists: sorted_spam, and sorted_ham. The files will show up  in this directories according to the algorithm you developed.

    test_spam_directory = 'sorted_spam_problem4'
    test_ham_directory = 'sorted_ham_problem4'

    if not os.path.exists(test_spam_directory):
        os.mkdir(test_spam_directory)
    if not os.path.exists(test_ham_directory):
        os.mkdir(test_ham_directory)

    dictionary_filename = "dictionary.dict"

    # create the dictionary to be used
    dictionary, spam_prior_probability = makedictionary(training_spam_directory, training_ham_directory,
                                                        dictionary_filename)
    # sort the mail
    spamsort(test_mail_directory, test_spam_directory, test_ham_directory, dictionary, spam_prior_probability)




if __name__ == "__main__":
    f = open('result.txt', 'wb')
    for i in range(50):
        result = cross_vali()
        f.write(str(result) + '\n')

