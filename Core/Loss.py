""" This file contains the loss functions and their derivatives. """


import numpy as np


class LossFunction:
    """
    A loss function contains two functions:
    1. A function that calculates the loss given the actual and predicted values.
    2. A function that calculates the derivative of the loss function given the actual and predicted values.
    """

    def loss(A: np.ndarray, y: np.ndarray):
        """
        :param A: what we actually predicted
        :param y: what we wanted
        :return: Returns a tuple. The first element of this tuple is the loss.
        Some loss functions may also return additional information.
        """
        pass

    def dLdA(A: np.ndarray, y: np.ndarray):
        """
        :param A: what we actually predicted
        :param y: what we wanted
        :return: the derivative of the loss function
        """
        pass


"""Squared Error Loss"""


class MSELoss(LossFunction):
    """
    Mean Squared Error Loss
    """

    def loss(A: np.ndarray, y: np.ndarray):
        return ((A - y) ** 2,)

    def dLdA(A, y):
        return 2 * (A - y)


class CategoricalCrossEntropy(LossFunction):
    """
    Categorical Cross Entropy
    """
    def loss(A, y):
        """
        :param A: what we actually predicted
        :param y: what we wanted
        :return: Returns a tuple. The first element of this tuple is the categorical cross entropy loss.
        The second element is a boolean array of the same shape as A, where the correct prediction is True.
        """
        return (- np.log(A[y][0]), A[y][0] == np.max(A))

    def dLdA(A, y):
        dLdA = np.zeros(A.shape)
        dLdA[y][0] = -1 / A[y][0]
        return dLdA


class BatchCategoricalCrossEntropy(LossFunction):
    """
    A batched version of Categorical Cross Entropy
    """
    def loss(A, y):
        # Compute loss
        loss = -np.log(A[y, np.arange(y.shape[0])])
        # Compute number of i such that A[y][i] = np.max(A[:, i])
        correct = np.count_nonzero(np.argmax(A, axis=0) == y)
        return (loss, correct)

    def dLdA(A, y):
        """
        :param A: what we actually predicted
        :param y: what we wanted
        :return:
        """

        dLdA = np.zeros(A.shape)

        # The following code is equivalent to:
        # for i in range(y.shape[0]):
        #     dLdA[y[i]][i] += -1 / A[y[i]][i]

        dLdA[y, np.arange(y.shape[0])] = -1 / A[y, np.arange(y.shape[0])]
        return dLdA
