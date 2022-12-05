import numpy as np

from Layer import Layer

from scipy.special import expit


def softmax(x):
    return np.exp(x - np.max(x)) / np.exp(x - np.max(x)).sum()


class Dense(Layer):
    def __init__(self, o_size: int, i_size: int = -1, activation: str = "none",
                 eta: float = 0.01):
        super().__init__()

        self.i_size: tuple = (i_size, 1)  # Redundant but okay

        self.o_size: tuple = (o_size, 1)

        self.f = activation

        self.eta = eta

        self.Z = None
        self.A = None
        self.Aout = None

        self.W = None
        self.W0 = None

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

        # m x n numpy array. Gaussian init.
        self.W = np.random.normal(0, 1 / self.i_size[0], (self.i_size[0], self.o_size[0]))
        self.W0 = np.random.normal(0, 1, self.o_size)

    def propagate(self, A):
        self.A = A  # Store input for backprop

        self.Z = np.matmul(self.W.T, self.A)

        self.Z += self.W0

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

    def dAdZ(self, dLdA):
        if self.f == "relu":
            dLdZ = np.diagflat(np.heaviside(self.Z, 0.5))
        elif self.f == "softmax":
            SM = self.Aout.reshape((-1, 1)) # Take Aout and reshape it to -1, 1
            dAdZ = np.diagflat(self.Aout) - np.dot(SM, SM.T)
            dLdZ = np.matmul(dAdZ, dLdA)
        elif self.f == "sigmoid":
            dLdZ = (self.Aout * (1 - self.Aout)) * dLdA
        elif self.f == "none":
            dLdZ = dLdA
        else:  # ignore
            dLdZ = dLdA
        return dLdZ


    def backpropagate(self, dLdA):
        dLdZ = self.dLdZ(dLdA)

        dLdW = np.matmul(self.A, dLdZ.T)
        dLdW0 = dLdZ

        self.W -= self.eta * dLdW
        self.W0 -= self.eta * dLdW0

        dLdA = np.matmul(self.W, dLdZ)

        return dLdA
