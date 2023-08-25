import math

import numpy as np

from Layer import Layer

import pickle

import time

from sys import stdout

"""How about for outputs in R."""


def sqloss(A, y):
<<<<<<< HEAD
    return (A - y) ** 2


def sqdLdA(A, y):
    return 2 * (A - y)
=======
    return (A - y) ** 2  # That was simple enough.


def sqdLdA(A, y):
    return 2 * (A - y)  # Lmao. Okay.
>>>>>>> master


"""These functions are valid for categorical crossentropy"""


def CCEcorrect(A, y):
    """
    :param A: what we actually predicted, e.g. a vector of 10 things for mnist.
    :param y: what we wanted, e.g. 2 or 4.
    :return:
    """

    return A[y][0] == np.max(A)


def CCEdLdA(A, y):
<<<<<<< HEAD
    dLdA = np.zeros(A.shape)
    dLdA[y][0] = -1 / A[y][0]
=======
    dLdA = np.zeros((10,), )  # TODO: Ugh get rid of hardcoded 10.
    dLdA[y] = -1 / A[y][0]
    dLdA = dLdA.reshape((10, 1))
>>>>>>> master
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

    return np.count_nonzero(np.argmax(A, axis=0) == y)  # Our predictions versus given


def CCEBatchdLdA(A, y):
    dLdA = np.zeros(A.shape)

    # TODO: Vectorize this?

    for i in range(y.shape[0]):
        dLdA[y[i]][i] += -1 / A[y[i]][i]  # Lol fine.
    return dLdA


def CCEBatchloss(A, y):
    """
    :param A: what we actually predicted
    :param y: what we wanted
    :return:
    """

    # ... lol?

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

    def test_forward(self, A):
        for layer in self.layers:
            A = layer.test(A)
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

<<<<<<< HEAD
=======

>>>>>>> master
    def test(self, x, y):
        """
        Test forward
        :param x:
        :return: result, failure (loss)
        """

        result = self.test_forward(x)
        failure = self.loss(result, y)
        return result, failure

    def save(self, filename=None):
<<<<<<< HEAD
        if filename is None:
=======
        if filename == None:
>>>>>>> master
            filename = "C://Users//rowec//PycharmProjects//learningML//Models//" + self.micro() + "-" + str(
                time.time_ns())
        pickle.dump(self, open(filename, "wb"))

    def train(self,
              x_train, y_train, x_test, y_test,
              correct,
              batch_size = None, test_batch_size = None):

        TRAIN_SIZE = x_train.shape[0]
        TEST_SIZE = x_test.shape[0]
        if batch_size is None:
            batch_size = TRAIN_SIZE
            batch_count = 1
        else:
            batch_count = TRAIN_SIZE // batch_size

        if test_batch_size is None:
            test_batch_size = TEST_SIZE
            test_batch_count = 1
        else:
            test_batch_count = TEST_SIZE // batch_size

        for epoch in range(batch_count):
            cumulative_loss = 0
            cumulative_correct = 0

            for i in range(batch_size):
                if(i % (batch_size // 100) == 0):
                    stdout.write("\r" + str(100 * i / batch_size)[:5] + "%")
                    stdout.flush()
                x = x_train[i + epoch * batch_size]
                y = y_train[i + epoch * batch_size]
                result, failure = self.cycle(x, y)
                cumulative_loss += failure

                cumulative_correct += correct(result, y)
            stdout.write("\r")
            print(f"Epoch = {epoch} ({batch_size} per batch) Average Loss = {cumulative_loss / batch_size}, Accuracy = {cumulative_correct / batch_size}")

            cumulative_loss = 0
            cumulative_correct = 0

            which_test_batch = epoch % test_batch_count

            for i in range(test_batch_size):
                if(i % (test_batch_size // 100) == 0):
                    stdout.write("\r" + str(100 * i / test_batch_size)[:5] + "%")
                    stdout.flush()
                x = x_test[i + which_test_batch * test_batch_size]
                y = y_test[i + which_test_batch * test_batch_size]
                result, failure = self.test(x, y)
                cumulative_loss += failure
                cumulative_correct += correct(result, y)
            stdout.write("\r")
            print(
                f"Testing --- Average Loss = {cumulative_loss / test_batch_size}, Accuracy = {cumulative_correct / test_batch_size}")
