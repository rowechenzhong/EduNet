import math

import numpy as np

#TODO: Make nice Model Class


def feed_forward(net, A):
    for layer in net:
        A = layer.propagate(A)
    return A

def loss(A, y):
    """
    :param A: what we actually predicted, i.e. (0.8, 0.1, 0.1, 0,0,00,0,0)
    :param y: what we wanted
    :return:
    """

    return - math.log(A[y][0])


def correct(A, y):
    """
    :param A: what we actually predicted, i.e. (0.8, 0.1, 0.1, 0,0,00,0,0)
    :param y: what we wanted
    :return:
    """

    return A[y][0] >= 0.1


def get_dLdA(A,y):
    dLdA = np.zeros((10,),)
    dLdA[y] = -1 / A[y][0]
    dLdA = dLdA.reshape((10, 1))
    return dLdA


def back_forward(net, dLdA):
    for layer in net[::-1]:
        dLdA = layer.backpropagate(dLdA)
    return dLdA