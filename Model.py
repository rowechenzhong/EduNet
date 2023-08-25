import math

import numpy as np

from Layer import Layer

import pickle

import time

from sys import stdout

from LossFunction import LossFunction


class Model:
    def __init__(self, loss: LossFunction, layers: list[Layer] = None):
        if layers is None:
            layers = []
        self.layers = layers

        self.lossFunction = loss

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
        dLdA = self.lossFunction.dLdA(result, y)
        self.back_forward(dLdA)

    def back_forward(self, dLdA):
        for layer in self.layers[::-1]:
            dLdA = layer.backpropagate(dLdA)
        return dLdA

    def cycle(self, x, y):
        """
        Feed forward then backpropagate
        :param x:
        :return: result, loss, additional information from loss function
        """
        result = self.feed_forward(x)
        failure = self.lossFunction.loss(result, y)
        dLdA = self.lossFunction.dLdA(result, y)
        self.back_forward(dLdA)

        return result, failure[0], failure[1:]

    def test(self, x, y):
        """
        Test forward
        :param x:
        :return: result, failure (loss), additional information from loss function
        """

        result = self.test_forward(x)
        failure = self.lossFunction.loss(result, y)
        return result, failure[0], failure[1:]

    def save(self, filename=None):
        if filename is None:
            filename = "C://Users//rowec//PycharmProjects//learningML//Models//" + self.micro() + "-" + str(
                time.time_ns())
        pickle.dump(self, open(filename, "wb"))

    def train(self,
              x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray, y_test: np.ndarray,
              batch_size=None, test_batch_size=None):

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
                if (i % (batch_size // 100) == 0):
                    stdout.write("\r" + str(100 * i / batch_size)[:5] + "%")
                    stdout.flush()
                x = x_train[i + epoch * batch_size]
                y = y_train[i + epoch * batch_size]
                result, failure, other_info = self.cycle(x, y)
                cumulative_loss += failure

                if len(other_info) > 0:
                    cumulative_correct += other_info[1]
            stdout.write("\r")
            print(
                f"Epoch = {epoch} ({batch_size} per batch)",
                f"Average Loss = {cumulative_loss / batch_size},",
                f"Accuracy = {cumulative_correct / batch_size}")

            cumulative_loss = 0
            cumulative_correct = 0

            which_test_batch = epoch % test_batch_count

            for i in range(test_batch_size):
                if (i % (test_batch_size // 100) == 0):
                    stdout.write(
                        "\r" + str(100 * i / test_batch_size)[:5] + "%")
                    stdout.flush()
                x = x_test[i + which_test_batch * test_batch_size]
                y = y_test[i + which_test_batch * test_batch_size]
                result, failure, other_info = self.test(x, y)
                cumulative_loss += failure
                if len(other_info) > 0:
                    cumulative_correct += other_info[1]
            stdout.write("\r")
            print(f"Testing --- Average Loss = {cumulative_loss / test_batch_size},",
                  f"Accuracy = {cumulative_correct / test_batch_size}")
