import math

import numpy as np

from Layer import Layer

import pickle

import time

"""How about for outputs in R."""
def sqloss(A, y):
    return (A - y)**2 #That was simple enough.

def sqdLdA(A, y):
    return 2 * (A - y) # Lmao. Okay.


"""These functions are valid for categorical crossentropy"""

def CCEcorrect(A, y):
    """
    :param A: what we actually predicted, e.g. a vector of 10 things for mnist.
    :param y: what we wanted, e.g. 2 or 4.
    :return:
    """

    return A[y][0] == np.max(A)


def CCEdLdA(A, y):
    dLdA = np.zeros(A.shape)
    dLdA[y][0] = -1 / A[y][0]
    return dLdA


def CCEloss(A, y):
    """
    :param A: what we actually predicted
    :param y: what we wanted
    :return:
    """

    return - np.log(A[y][0])


def CCEBatchcorrect(A, y):
    """
    :param A: what we actually predicted, e.g. a vector of 10 things for mnist.
    :param y: what we wanted, e.g. 2 or 4.
    :return:
    """

    return np.count_nonzero(np.argmax(A, axis=0) == y) # Our predictions versus given

def CCEBatchdLdA(A, y):
    dLdA = np.zeros(A.shape)

    #TODO: Vectorize this?

    for i in range(y.shape[0]):
        dLdA[y[i]][i] += -1/A[y[i]][i] # Lol fine.
    return dLdA


def CCEBatchloss(A, y):
    """
    :param A: what we actually predicted
    :param y: what we wanted
    :return:
    """

    #... lol?

    return np.sum(-np.log(A[y, np.arange(y.shape[0])]))


class Model:
    def __init__(self, loss, dLdA, layers: list[Layer] = None):
        if layers is None:
            layers = []
        self.layers = layers

        self.loss = loss
        self.dLdA = dLdA

    def micro(self):
        """
        Returns short string representation for filename purposes.
        :return:
        """
        return "".join(layer.micro() for layer in self.layers)

    def __str__(self):
        res = f"Model, Layers = {len(self.layers)}\n"
        for i in self.layers:
            res += str(i) + "\n"
        return res

    def compile(self):
        """
        Calculates implicit inputs for intermediate layers.
        :return:
        """
        for i in range(len(self.layers) - 1):
            self.layers[i + 1].update_input(self.layers[i].get_output())

    def join(self, layer: Layer):
        self.layers.append(layer)

    def feed_forward(self, A):
        for layer in self.layers:
            A = layer.propagate(A)
        return A

    def feed_backward(self, result, y):
        dLdA = self.dLdA(result, y)
        self.back_forward(dLdA)

    def back_forward(self, dLdA):
        for layer in self.layers[::-1]:
            dLdA = layer.backpropagate(dLdA)
        return dLdA

    def cycle(self, x, y):
        """
        Feed forward then backpropagate
        :param x:
        :return: result, loss
        """
        result = self.feed_forward(x)
        failure = self.loss(result, y)
        dLdA = self.dLdA(result, y)
        self.back_forward(dLdA)

        return result, failure



    def test(self, x, y):
        """
        Feed forward
        :param x:
        :return: result, failure (loss)
        """

        result = self.feed_forward(x)
        failure = self.loss(result, y)
        return result, failure

    def save(self, filename = None):
        if filename == None:
            filename = "C://Users//rowec//PycharmProjects//learningML//Models//" + self.micro() + "-" + str(time.time_ns())
        pickle.dump(self, open(filename, "wb"))
