
import numpy as np


class Perceptron:

    def __init__(self, inputs, zeros=True):
        '''
        Contructor gives perceptron zeros for weights and bias
        :param inputs: number of dimensions in feature vectors
        :param zeros: if true then weights and bias are set to zero, otherwise they are assigned random values
        '''
        if zeros:
            self.weights = np.zeros(inputs)
            self.bias = np.zeros(1)
        else:
            self.weights = np.random.uniform(low=0, high=1, size=inputs)
            self.bias = np.random.uniform(low=0, high=1)


    def activation(self, x):
        '''
        Activation function (weights DOT x) + bias. Returns 1 or 0 based on positive or negative ans.
        :param x: feature vector
        :return: -1 or 1 based on return of activation
        '''
        return 1 if np.dot(self.weights, x) + self.bias > 0 else -1



    def train(self, data, epochs=1000, c=1):
        '''
        Generator Function.
        Yields a series of tuples, each containing alist of weights and bias for each increment of the perceptron.
        Based on Rosenblatt's single layer perceptron algorithm
        :param features: numpy array of feature vectors
        :param labels: numpy array of labels
        :param epochs: number of training iterations for the perceptron
        :param c: constant multiplier for increments of weight vecotr
        :yields array of weights and a bias constant for each iteration of perceptron
        :return: number of iterations to convergence
        '''
        error_detected = True
        itr = 0
        while error_detected and itr < epochs:
            itr += 1
            error_detected = False
            for points in data:
                x = points.features.toArray()
                lab = points.label
                if self.activation(x) != lab:
                    error_detected = True
                    self.weights += c*(x * lab)
                    self.bias += c*lab
            # yield self.weights, self.bias
        return itr

    def predict(self, points):
        return [self.activation(x) for x in points]

    def test_error(self, features, labels):
        error_count = 0
        for i, x in enumerate(features):
            if self.activation(x) != labels[i]:
                error_count += 1
        return error_count/len(features)

