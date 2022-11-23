import math

import numpy as np

from Layer import Layer

import pickle

import time


def correct(A, y):
    """
    :param A: what we actually predicted, i.e. (0.8, 0.1, 0.1, 0,0,00,0,0)
    :param y: what we wanted
    :return:
    """

    return A[y][0] == np.max(A)


def get_dLdA(A, y):
    dLdA = np.zeros((10,), )
    dLdA[y] = -1 / A[y][0]
    dLdA = dLdA.reshape((10, 1))
    return dLdA


def loss(A, y):
    """
    :param A: what we actually predicted, i.e. (0.8, 0.1, 0.1, 0,0,00,0,0)
    :param y: what we wanted
    :return:
    """

    return - math.log(A[y][0])


class Model:
    def __init__(self, layers: list[Layer] = None):
        if layers is None:
            layers = []
        self.layers = layers

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
        failure = loss(result, y)
        dLdA = get_dLdA(result, y)
        self.back_forward(dLdA)

        return result, failure

    def test(self, x, y):
        """
        Feed forward
        :param x:
        :return: result, loss
        """

        result = self.feed_forward(x)
        failure = loss(result, y)
        return result, failure

    def save(self, filename = None):
        if filename == None:
            filename = "C://Users//rowec//PycharmProjects//learningML//Models//" + self.micro() + str(time.time_ns())
        pickle.dump(self, open(filename, "wb"))
