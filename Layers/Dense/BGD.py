import numpy as np

from Core.Layer import Layer

from scipy.special import expit

from Util.AdamArray import *


def softmax(x):
    return np.exp(x - np.max(x)) / np.exp(x - np.max(x)).sum(axis=0)


class Dense(Layer):
    def __init__(self, o_size: int, i_size: int = -1, activation: str = "none"):
        super().__init__()

        self.i_size: tuple = (i_size, 1)  # Redundant but okay

        self.o_size: tuple = (o_size, 1)

        self.f = activation

        self.Z = None
        self.A = None
        self.Aout = None

        if i_size != -1:
            self.update_input(i_size)

    def __str__(self):
        return f"Dense Layer {self.i_size[0]} -> {self.o_size[0]}, {self.f} activation"

    def micro(self):
        return f"D{self.f[:2]}{self.o_size[0]}"

    def update_input(self, i_size):
        if type(i_size) == int:
            self.i_size = (i_size, 1)
        elif type(i_size) == tuple:
            self.i_size = i_size

        self.W = AdamArray((self.i_size[0], self.o_size[0]))
        self.W0 = AdamArray(self.o_size)

    def propagate(self, A):
        self.A = A  # Store input for backprop

        self.Z = np.matmul(self.W.W.T, self.A)

        self.Z += self.W0.W

        if self.f == "relu":
            self.Aout = np.maximum(0, self.Z)
        elif self.f == "softmax":
            self.Aout = softmax(self.Z)
        elif self.f == "sigmoid":
            self.Aout = expit(self.Z)
        elif self.f == "none":
            self.Aout = self.Z
        else:
            self.Aout = self.Z

        return self.Aout

    def backpropagate(self, dLdA):
        if self.f == "relu":
            dLdZ = np.heaviside(self.Z, 0.5) * dLdA
        elif self.f == "softmax":
            dLdZ = self.Aout * (dLdA - np.diagonal(self.Aout.T @ dLdA))
        elif self.f == "sigmoid":
            dLdZ = (self.Aout * (1 - self.Aout)) * dLdA
        elif self.f == "none":
            dLdZ = dLdA
        else:  # ignore
            dLdZ = dLdA

        dLdW = np.matmul(self.A, dLdZ.T)
        dLdW0 = np.sum(dLdZ, axis=1).reshape((-1, 1))

        dLdA = np.matmul(self.W.W, dLdZ)

        self.W.update(dLdW)
        self.W0.update(dLdW0)
        return dLdA
