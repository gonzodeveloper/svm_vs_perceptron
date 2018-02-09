
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
            self.weights = np.random.uniform(low=-1, high=1, size=inputs)
            self.bias = np.random.uniform(low=-1, high=1)

    def activation(self, x):
        '''
        Activation function (weights DOT x) + bias. Returns 1 or -1 based on positive or negative ans.
        :param x: feature vector
        :return: -1 or 1 based on return of activation
        '''
        return 1 if np.dot(self.weights, x) + self.bias > 0 else -1

    def train(self, points, labels, epochs=1000, c=1):
        '''
        Train our perceptron with a list of points and their labels
        :param points: list points for training (each should be np.array)
        :param labels: list of labels (either 1 or -1)
        :param epochs: limit number of epochs
        :param c: learning rate
        :return: iterations required to convergence
        '''
        error_detected = True
        itr = 0
        while error_detected and itr < epochs:
            itr += 1
            error_detected = False
            for i, x in enumerate(points):
                if self.activation(x) != labels[i]:
                    error_detected = True
                    self.weights += c*(x * labels[i])
                    self.bias += c*labels[i]
            # yield self.weights, self.bias
        return itr

    def predict(self, points):
        '''
        Predict the labels for a list of points
        :param points: list of points (each point is np.array)
        :return: list of labels
        '''
        return [self.activation(x) for x in points]


