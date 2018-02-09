from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
# from perceptronv3 import Perceptron
from sklearn.linear_model.perceptron import Perceptron
import numpy as np
import pandas as pd
from multiprocessing import Pool



class Testor:

    def __init__(self, low, high, step_size, runs):
        self.low = low
        self.high = high
        self.runs = runs
        self.total = runs * ((high - low)/step_size) + runs
        self.iteration = 0

    def dist_from_hyplane(self, x, w, b):
        '''
        Get the distance of a point x from they hyperplane defined by its intersect and normal
        :param x: point
        :param w: normal vector
        :param b: intersect
        :return: distance
        '''
        return (np.dot(w, x) + b)/np.linalg.norm(w)

    def generate_labeled_points(self, n, dim):
        '''
        Generate a set on n linearly separable points with a given dimensionality
        :param n: number of points
        :param dim: dimensionality of points
        :return: list of points (np arrays) and their labels 1 or -1, and gamma (i.e., the margin)
        '''

        # Define a random hyperplane in space
        norm = np.random.uniform(low=-1, high=1, size=dim)
        intercept = 0   # np.random.uniform(low=-1, high=1)

        # Flags, we need to ensure that we have at least one point with each label
        has_neg = False
        has_pos = False
        i = 0

        gammas = []
        data = []
        while i < n:
            # Get point, its distance from hyperplane
            point = np.random.uniform(low=-1, high=1, size=dim)
            dist = self.dist_from_hyplane(point, norm, intercept)
            gammas.append(abs(dist))
            # Determine label
            label = 1 if dist > 0 else -1
            data.append([point, label])
            i += 1

            # Making sure that we have at least one point of each type
            if np.sign(dist) > 0:
                has_pos = True
            else:
                has_neg = True
            if not (has_pos and has_neg) and i == n:
                i -= 1
                data.pop()
        # Get minimum distance of any points from the plane. this is the margin
        gamma = min(gammas)
        return data, gamma

    def test_error(self, predictions, labels):
        '''
        Figure out float valued test error from a list of predictions and labels
        :param predictions: list of predicted classifications
        :param labels: list of actual labels
        :return: flaot valued error 0-1
        '''
        error_count = 0
        total = len(predictions)
        for x, y in zip(predictions, labels):
            if x != y:
                error_count += 1
        return error_count/total

    def simulation(self, n, dim=2):
        '''
        Run a simulation to test svm and perceptron on a given amount of points. Each run of the simulation generates a set
        of data then trains a perceptron and a svm with a given slice and tests both of their predictions' for accuracy.
        A csv is written where each row represents a run. The row contains data on the number of points, margin of data,
        perceptron weights, svm weighs, perceptron prediction error, svm prediction error, and ratio of svm/perceptron weight
        megnitudes for that run,
        :param test_range: range of n to test on
        :param step_size: step size to increment n
        :param file: file name for write
        :param runs: number of runs per n step
        :param dim: dimensionality of data
        :return:
        '''
        # For progress bar
        prefix = "Simulation"
        sufix = "Complete"

        all_data = []
        for i in range(self.runs):
            # Which iteration out of total are we on?
            self. iteration += 1
            self.print_progress_bar(self.iteration, self.total, prefix, sufix)

            # Get test data and its gamma, split 80-20 test train
            data, gamma = self.generate_labeled_points(n, dim)
            train_dat, test_dat = train_test_split(data, test_size=.2)

            # Separate train points from labels
            train_points = [x[0] for x in train_dat]
            train_labels = [x[1] for x in train_dat]

            # Separate test points from their labels
            test_points = [x[0] for x in test_dat]
            test_labels = [x[1] for x in test_dat]

            # Train and test with perceptron
            perceptron = Perceptron()
            perceptron.fit(train_points, train_labels)
            predictions = perceptron.predict(test_points)
            p_error = self.test_error(predictions, test_labels)

            dist = []
            for x in train_points:
                dist.append(abs(self.dist_from_hyplane(x, perceptron.coef_, perceptron.intercept_)))
            p_gamma = float(min(dist))

            # Train and test with SVM
            svm = SVC(kernel="linear")
            # Throw out the rare case where our train is all same class
            try:
                svm.fit(train_points, train_labels)
            except ValueError:
                continue
            predictions = svm.predict(test_points)
            svm_error = self.test_error(predictions, test_labels)

            dist = []
            for x in train_points:
                dist.append(abs(self.dist_from_hyplane(x, svm.coef_, svm.intercept_)))
            s_gamma = float(min(dist))

            all_data.append([n, gamma, p_gamma, s_gamma,  p_error, svm_error])

        df = pd.DataFrame(all_data, columns=['n', 'gamma', 'perceptron gamma', 'svm gamma',
                                       'perceptron_error', 'svm_error'])

        return df

    def print_progress_bar (self, iteration, total, prefix='', suffix='', decimals=2, length=50, fill='█'):
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
    # simulation(test_range=(100,100000),step_size=100, runs=100, file="data/run5.csv")

    testor = Testor(low=100, high=1000, step_size=100, runs=40)
    tasks = []
    for n in range(100, 100001, 100):
        tasks.append((n,))

    with Pool(processes=4) as pool:
        results = [pool.apply_async(testor.simulation, args=t) for t in tasks]
        data = [r.get() for r in results]

    df = pd.concat(data)
    df.to_csv("data/multi.csv", sep=',', index=False)
