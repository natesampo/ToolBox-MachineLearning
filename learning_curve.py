""" Exploring learning curves for classification of handwritten digits """

import matplotlib.pyplot as plt
import numpy
from sklearn.datasets import *
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LogisticRegression


def display_digits():
    digits = load_digits()
    print(digits.DESCR)
    fig = plt.figure()
    for i in range(10):
        subplot = fig.add_subplot(5, 2, i+1)
        subplot.matshow(numpy.reshape(digits.data[i], (8, 8)), cmap='gray')

    plt.show()


def train_model():
    data = load_digits()
    num_trials = 100
    train_percentages = range(5, 95, 5)
    test_accuracies = numpy.zeros(len(train_percentages))
    for train_set in train_percentages:
        model_score = 0
        for i in range(0, num_trials):
            X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, train_size=train_set)
            model = LogisticRegression(C=10**-10)
            model.fit(X_train, y_train)
            model_score += model.score(X_test, y_test)
        test_accuracies[int(train_set/5)-1] = model_score/num_trials
        '''print("Train accuracy %f" %model.score(X_train, y_train))
        print("Test accuracy %f"%model.score(X_test, y_test))'''
    fig = plt.figure()
    plt.plot(train_percentages, test_accuracies)
    plt.xlabel('Percentage of Data Used for Training')
    plt.ylabel('Accuracy on Test Set')
    plt.show()


if __name__ == "__main__":
    # Feel free to comment/uncomment as needed
    #display_digits()
    train_model()
