from sklearn.model_selection import train_test_split
from pyspark import SparkContext
from pyspark.mllib.classification import SVMWithSGD
from pyspark.mllib.regression import LabeledPoint
from pyspark.sql import SparkSession
from perceptronv2 import Perceptron
import numpy as np
import os
import pandas as pd


def dist_from_hyplane(x, w, b):
    return (np.dot(w, x) - b)/np.linalg.norm(w)


def generate_labeled_points(n, dim):

    norm = np.random.uniform(low=0, high=1, size=dim)
    intercept = np.random.uniform(low=0, high=1)
    gamma = np.random.uniform(low=0, high=1)

    points = []
    count = 0
    while count < n:
        point = np.random.uniform(low=0, high=1, size=dim)

        dist = dist_from_hyplane(point, norm, intercept)
        # Throw out point in inside margin
        if abs(dist) >= gamma:
            gamma = min(gamma, dist)
            label = 1 if dist >= 0 else 0
            points.append(LabeledPoint(label,point))
            count += 1

    return points, gamma


def test_error(predictions, labels):
    error_count = 0
    total = len(predictions)
    for x, y in zip(predictions, labels):
        if x != y:
            error_count += 1
    return error_count/total


def simulation(test_range, step_size, file, runs=100, dim=2):
    '''

    :param test_range:
    :param step_size:
    :param file:
    :param runs:
    :param dim:
    :return:
    '''
    prefix = "Simulation"
    sufix = "Complete"

    (low, high) = test_range
    total = runs * ((high - low)/step_size)
    n = low
    data = []
    df = pd.DataFrame()
    while n <= high:
        for i in range(runs):
            points, gamma = generate_labeled_points(n, dim)
            train_dat, test_dat = train_test_split(np.array(points), test_size=.2)

            test_points = [x.features.toArray() for x in test_dat]
            test_labels = [x.label for x in test_dat]

            perceptron = Perceptron(dim)
            perceptron.train(train_dat)
            predictions = perceptron.predict(test_points)
            p_error = test_error(predictions, test_labels)

            svm = SVMWithSGD.train(sc.parallelize(train_dat))
            predictions = svm.predict(sc.parallelize(test_points))
            svm_error = test_error(predictions.collect(), test_labels)

            data.append([n, gamma, perceptron.weights, p_error, svm.weights, svm_error])
            print(data[i])
        n += step_size

    df = pd.DataFrame(data, columns=['n', 'gamma', 'perceptron_weights', 'perceptron_error', 'svm_weights', 'svm_error'])
    # df.to_csv(file, sep=',', index=False)

    return df


def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 2, length = 50, fill = '█'):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
    """
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

    df = simulation((10,100),10, file=None)
    df
