#!/usr/bin/env python

import os,sys
import csv
import numpy
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import random

from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_iris
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import GridSearchCV


class MidpointNormalize(Normalize):

    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y))


#TOTAL_SAMPLES = [100, 200, 500, 1000, 2000, 5000, 10000, 20000, 50000]
TOTAL_SAMPLES = 5000


def main():
    #size = int(sys.argv[1])

    train = 0.6 * TOTAL_SAMPLES
    cv = 0.8 * TOTAL_SAMPLES
    test = 1 * TOTAL_SAMPLES

    train_cv = int(train+cv)

    # Load and prepare data set, do not need pre-processing, directly use the dataset
    data = open('../data/training.csv', "rb")
    reader = csv.reader(data, delimiter = ',', quoting=csv.QUOTE_NONE)

    x = list(reader)
    data = numpy.array(x).astype('string')

    # #############################################################################
    # Train classifiers
    # Now we need to fit a classifier for all parameters

    C_range = [0.1, 1, 10, 1e2, 1e3, 1e4, 1e5]
    gamma_range = [1e-7, 5e-7, 1e-6, 5e-6, 1e-5, 5e-5, 1e-4, 5e-4]

    #C_range = [1000]
    #gamma_range = [5e-5]

    classifiers = []

    scores_train = []
    scores_cv = []

    max_accuracy = 0

    raw_dataset = []
    choice = []

    # initialize the data index, with TOTAL_SAMPLES defined above
    for i in range(1, TOTAL_SAMPLES+1):
        raw_dataset.append(i)

    #shuffle the whole dataset to make it fair distributed
    random.shuffle(raw_dataset)

    # Test data is reserved and never touched for the last 20% of the data
    X_30dtestraw = data[raw_dataset[int(cv) + 1 : int(test)], 1:31]
    X_30dtest = X_30dtestraw.astype('float')
    Y_30dtest = data[raw_dataset[int(cv) + 1 : int(test)], 32]

    # Prepare the dataset for training & cross-validation for the first 80% of the data
    for i in range(0, int(cv)):
        choice.append(raw_dataset[i])

    for i in range(0, len(C_range)):
        Cscores_train = []
        Cscores_cv = []

        for j in range(0, len(gamma_range)):
            raw_score_train = []
            raw_score_cv = []
            raw_precision = []
            raw_recall = []
            '''
            Cross-Validation 5 times by randomly shuffle data points
            '''
            for cv_iteration in range(0, 5):
                #shuffle data points here for each round
                random.shuffle(choice)

                #Randomly generate training data
                X_30draw = data[choice[0:int(train)], 1:31]
                X_30d = X_30draw.astype('float')
                Y_30d = data[choice[0:int(train)], 32]

                #Randomly generate cross-validation data
                X_30dcvraw = data[choice[int(train): int(cv)], 1:31]
                X_30dcv = X_30dcvraw.astype('float')
                Y_30dcv = data[choice[int(train): int(cv)], 32]

                clf = SVC(C=C_range[i], gamma=gamma_range[j])
                clf.fit(X_30d, Y_30d)
                classifiers.append((C_range[i], gamma_range[j], clf))
                res_train = clf.predict(X_30d)
                res_cv = clf.predict(X_30dcv)

                count_train = 0

                # Get statistics of accuracy of training
                for m in range(0, len(Y_30d)):
                    if str(res_train[m]) == str(Y_30d[m]):
                        count_train += 1
                raw_score_train.append(count_train / train)

                # Initial true positive, true negative, false positive and false negative to 0
                tp = 0
                tn = 0
                fp = 0
                fn = 0

                # Get statistics of accuracy of cross-validation
                for m in range(0,len(Y_30dcv)):
                    if str(res_cv[m]) == 's' and str(Y_30dcv[m]) == 's':
                        tp += 1
                    elif str(res_cv[m]) == 's' and str(Y_30dcv[m]) == 'b':
                        fp += 1
                    elif str(res_cv[m]) == 'b' and str(Y_30dcv[m]) == 's':
                        fn += 1
                    else:
                        tn += 1

                count_cv = tp+tn

                raw_score_cv.append(count_cv/(cv-train))

                raw_precision.append(float(tp)/float(tp+fp))
                raw_recall.append(float(tp)/float(tp+fn))

            # Document the best hyperparameter so far, and run the test set
            if count_cv > max_accuracy:
                max_accuracy = count_cv
                res_test = clf.predict(X_30dtest)
                count_test = 0
                for m in range(0, len(Y_30dtest)):
                    if str(res_test[m]) == str(Y_30dtest[m]):
                        count_test += 1

                max_com = (i, j, count_test / (test-cv))

            Cscores_train.append(numpy.mean(raw_score_train))
            Cscores_cv.append(numpy.mean(raw_score_cv))
            precision = numpy.mean(raw_precision)
            recall = numpy.mean(raw_recall)
            f1 = float(2*precision*recall/(precision+recall))

            print 'Training accuracy of C= {} beta = {} is {}'.format(C_range[i], gamma_range[j], numpy.mean(raw_score_train))
            print 'Cross-validation accuracy of C= {} beta = {} is {}'.format(C_range[i], gamma_range[j], numpy.mean(raw_score_cv))
            print 'precision = {}, recall = {}, f1 = {}'.format(precision, recall, f1)

        scores_train.append(Cscores_train)
        scores_cv.append(Cscores_cv)

    print "The hyperparameter(s) that gives the highest cross-validation accuracy is C = {} beta = {}, with test accuracy of {}".format(C_range[max_com[0]], gamma_range[max_com[1]], float(max_com[2]))


    '''
    # Use as needed
    # Draw the graph of the validation accuracy as a function of beta or C
    #choose single value C and multiple value gamma or single value gamma and multiple value Csss
    plt.figure(figsize=(8,6))
    plt.xlabel('C')
    #plt.xlabel('beta')
    plt.ylabel('Accuracy')
    plt.xscale('log')
    plt.plot(C_range, scores_train, c='red', linewidth=2.0, label='training accuracy')
    plt.plot(C_range, scores_cv, c='blue', linewidth=2.0, label='cross-validation accuracy')
    #plt.plot(gamma_range, scores_train, c='red', linewidth=2.0, label='training accuracy')
    #plt.plot(gamma_range, scores_cv, c='blue', linewidth=2.0, label='cross-validation accuracy')
    plt.legend()
    plt.savefig("cv_C")
    plt.show()
    '''

    # Draw heatmap of the validation accuracy as a function of gamma and C
    #
    # The score are encoded as colors with the hot colormap which varies from dark
    # red to bright yellow. As the most interesting scores are all located in the
    # 0.6 to 0.8 range we use a custom normalizer to set the mid-point to 0.7 so
    # as to make it easier to visualize the small variations of score values in the
    # interesting range while not brutally collapsing all the low score values to
    # the same color.

    plt.figure(figsize=(8, 6))
    plt.subplots_adjust(left=.2, right=0.95, bottom=0.15, top=0.95)
    plt.imshow(scores_cv, interpolation='nearest', cmap=plt.cm.hot,
           norm=MidpointNormalize(vmin=0.6, midpoint=0.70))
    plt.xlabel('beta')
    plt.ylabel('C')
    plt.colorbar()
    plt.xticks(np.arange(len(gamma_range)), gamma_range, rotation=45)
    plt.yticks(np.arange(len(C_range)), C_range)
    plt.title('Validation accuracy')
    plt.show()

if __name__ == '__main__':
    main()
