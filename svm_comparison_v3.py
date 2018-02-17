from sklearn.svm import SVC
from sklearn.linear_model.perceptron import Perceptron
from math import ceil
import numpy as np
import pandas as pd
from multiprocessing import Pool


def dist_from_hyplane(x, w, b):
    '''
    Get the distance of a point x from they hyperplane defined by its intersect and normal
    :param x: point
    :param w: normal vector
    :param b: intersect
    :return: distance
    '''
    return (np.dot(w, x) + b) / np.linalg.norm(w)


def generate_labeled_points(n_train, n_test, dim, gamma=0):
    '''
    Generate a random set of linearly seperable training and testing points in the given dimension, ensuring
    training points
    :param n_train: number of training points; int
    :param n_test: number of testing points; int
    :param dim: dimensionality of points
    :return: np array: linearly separable train points, np array: linearly separable test points
    '''
    # Define a random hyperplane in space
    norm = np.random.uniform(low=-1, high=1, size=dim)
    intercept = 0  # np.random.uniform(low=-1, high=1)

    # Flags, we need to ensure that we have at least one train point with each label
    has_neg = False
    has_pos = False
    i = 0

    train = []
    gammas = []
    while i < n_train:
        # Get point, label and its distance from hyperplane
        point = np.random.uniform(low=-1, high=1, size=dim)
        dist = dist_from_hyplane(point, norm, intercept)
        label = 1 if dist > 0 else -1

        # Update flags to show that we have pos or neg point
        if np.sign(dist) > 0:
            has_pos = True
        else:
            has_neg = True

        # If we get to the end and we do not have one of each point, try again
        if not (has_pos and has_neg) and i == n_train:
            continue
        # If point violates gamma, try again
        if abs(dist) < gamma:
            continue

        train.append([point, label])
        gammas.append(abs(dist))
        i += 1

    # Get minimum distance of any points from the plane. this is the margin
    margin = min(gammas)

    test = []
    for x in range(n_test):
        # Get point, its distance from hyperplane
        point = np.random.uniform(low=-1, high=1, size=dim)
        dist = dist_from_hyplane(point, norm, intercept)
        # Determine label
        label = 1 if dist > 0 else -1
        test.append([point, label])

    return train, test, margin


def simulation(n, runs, margin=0, p_runs=100, d=2):
    '''
    Run a a given number of simulations to compare svm and perceptron error rates. Generates a set of training and testing points, runs a
    single svm and a given number of perceptrons (avg error is taken)
    :param n: number of points
    :param p_runs: number of perceptrons to average
    :param runs: number of times to sun simulation
    :param d: dimensionality of points
    :return: pandas dataframe, each row
    '''

    all_data = []
    for i in range(runs):
        # Get test data and its gamma, split 80-20 test train
        train_dat, test_dat, margin = generate_labeled_points(n_train=n, n_test=ceil(n * 25), gamma=margin, dim=d)

        # Separate train points from labels
        train_points = [x[0] for x in train_dat]
        train_labels = [x[1] for x in train_dat]

        # Separate test points from their labels
        test_points = [x[0] for x in test_dat]
        test_labels = [x[1] for x in test_dat]

        # Run k = p_runs number of perceptrons on this same training data, take their mean error
        p_errors = []
        seed = np.random.RandomState()
        for k in range(p_runs):
            perceptron = Perceptron(random_state=seed)
            perceptron.fit(train_points, train_labels)
            p_errors.append(perceptron.score(test_points, test_labels))
        p_error = np.mean(p_errors)

        # Train and test with single SVM
        svm = SVC(kernel="linear")
        svm.fit(train_points, train_labels)
        svm_error = svm.score(test_points, test_labels)

        all_data.append([n, margin, p_error, svm_error])

    df = pd.DataFrame(all_data, columns=['n', 'margin', 'avg perceptron_error', 'svm_error'])
    return df


def print_progress_bar (iteration, total, prefix='', suffix='', decimals=2, length=50, fill='█'):
    '''
    Auxillary function. Gives us a progress bar which tracks the completion status of our task. Put in loop.
    :param iteration: current iteration
    :param total: total number of iterations
    :param prefix: string
    :param suffix: string
    :param decimals: float point precision of % done number
    :param length: length of bar
    :param fill: fill of bar
    :return:
    '''
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end = '\r')
    # Print New Line on Complete
    if iteration == total:
        print()


if __name__ == "__main__":

    '''
    # Increase n by exponents of 2: start at n=2**1, end at n=2**16=65,536
    # For each n we run simulation on 100 randomly generated sets of points
    lo = 1
    hi = 16
    run = 100
    '''

    # Fix n to 100 points
    n = 100
    run = 100
    # Vary gamma from 0.1 - 1
    lo = 0.01
    hi = 1

    # Progress bar stuff
    iteration = 0
    total = 50 * 100
    prefix = "Simulation"
    suffix = "Complete"
    print_progress_bar(iteration, total, prefix=prefix, suffix=suffix)


    tasks = []
    for g in map(lambda x: x/100.0, range(1, 100, 1)):
        tasks.append((n, run, g, ))

    # Send our tasks to the process pool, as they complete append their results to data
    data = []
    with Pool(processes=3) as pool:
        results = [pool.apply_async(simulation, args=t) for t in tasks]
        for r in results:
            iteration += run
            data.append(r.get())
            print_progress_bar(iteration, total, prefix=prefix, suffix=suffix)

    print("Writing data...")
    df = pd.concat(data)
    df.to_csv("data/small.csv", sep=',', index=False)

