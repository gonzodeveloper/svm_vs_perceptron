from sklearn.model_selection import train_test_split
from pyspark import SparkContext
from pyspark.mllib.classification import SVMWithSGD
from pyspark.mllib.regression import LabeledPoint
from pyspark.sql import SparkSession
from perceptronv2 import Perceptron
import numpy as np
import os
import pandas as pd


def svm_labels(label):
    return 1 if label == 1 else 0


def dist_from_hyplane(x, w, b):
    return (np.dot(w, x) + b)/np.linalg.norm(w)


def generate_labeled_points(n, dim):

    norm = np.random.uniform(low=-1, high=1, size=dim)
    intercept = np.random.uniform(low=-1, high=1)
    gammas = []

    points = []
    for i in range(n):
        point = np.random.uniform(low=-1, high=1, size=dim)

        dist = dist_from_hyplane(point, norm, intercept)
        # Throw out point in inside margin
        gammas.append(abs(dist))
        label = 1 if dist > 0 else -1
        points.append(LabeledPoint(label,point))

    gamma = min(gammas)
    return points, gamma


def test_error(predictions, labels):
    error_count = 0
    total = len(predictions)
    for x, y in zip(predictions, labels):
        if x != y:
            error_count += 1
    return error_count/total


def simulation(test_range, step_size, file, runs=100, dim=2):

    (low, high) = test_range
    # For progress bar
    prefix = "Simulation"
    sufix = "Complete"
    total = runs * ((high - low)/step_size) + runs
    iteration = 0

    data = []
    for n in range(low, high+1, step_size):
        for i in range(runs):
            # Which iteration out of total are we on?
            iteration += 1
            print_progress_bar(iteration, total, prefix, sufix)

            points, gamma = generate_labeled_points(n, dim)
            # 80/20 train-test split
            train_dat, test_dat = train_test_split(np.array(points), test_size=.2)
            # Relabel to 1, 0 for SVM
            svm_train_dat = np.array([LabeledPoint(svm_labels(x.label),x.features) for x in train_dat])

            # Separate test points from their labels
            test_points = [x.features.toArray() for x in test_dat]
            test_labels = [x.label for x in test_dat]
            # Relabel to 1, 0 for SVM
            svm_test_labels = np.array([svm_labels(x.label) for x in test_dat])

            # Train and test with perceptron
            perceptron = Perceptron(dim, zeros=False)
            perceptron.train(train_dat)
            predictions = perceptron.predict(test_points)

            p_error = test_error(predictions, test_labels)

            # Train and test with SVM
            svm = SVMWithSGD.train(sc.parallelize(svm_train_dat), regType=None)
            predictions = svm.predict(sc.parallelize(test_points))
            svm_error = test_error(predictions.collect(), svm_test_labels)

            # Get magnitudes of perceptron and svm final weight vectors
            p_len = np.linalg.norm(perceptron.weights)
            svm_len = np.linalg.norm(svm.weights.toArray())
            data.append([n, gamma, perceptron.weights, svm.weights.toArray(), p_error, svm_error, (svm_len/p_len)])

    df = pd.DataFrame(data, columns=['n', 'gamma', 'perceptron_weights',
                                     'svm_weights', 'perceptron_error', 'svm_error', 'svm/percp'])
    df.to_csv(file, sep=',', index=False)


def print_progress_bar (iteration, total, prefix = '', suffix = '', decimals = 2, length = 50, fill = '█'):
    
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end = '\r')
    # Print New Line on Complete
    if iteration == total:
        print()


if __name__ == "__main__":
    # Get Spark context
    print("Configuring Spark... \n")
    os.environ['PYSPARK_PYTHON'] = '/usr/bin/python3'
    sc = SparkContext("local[4]", "svm")
    sc.setLogLevel(logLevel="OFF")
    spark = SparkSession(sparkContext=sc)


    simulation(test_range=(5,1000),step_size=5,runs=100, file="data/run1.csv")
