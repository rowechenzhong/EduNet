import numpy as np

from Layer import Layer

from scipy.special import expit


def softmax(x):
    return np.exp(x - np.max(x)) / np.exp(x - np.max(x)).sum(axis=0)


class Dense(Layer):
    def __init__(self, o_size: int, i_size: int = -1, activation: str = "none",
                 eta: float = 0.01, t0: float = 1, dt: float = 0.0001):
        super().__init__()

        self.i_size: tuple = (i_size, 1)  # Redundant but okay

        self.o_size: tuple = (o_size, 1)

        self.f = activation

        self.eta = eta

        self.t = t0
        self.dt = dt

        self.B1 = 0.9
        self.B2 = 0.999
        self.eps = 10 ** -8

        self.Z = None
        self.A = None
        self.Aout = None

        self.W = None
        self.W0 = None
        self.m = None
        self.m0 = None
        self.v = None
        self.v0 = None

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

        self.m = np.zeros(self.W.shape)
        self.m0 = np.zeros(self.W0.shape)
        self.v = np.zeros(self.W.shape)
        self.v0 = np.zeros(self.W0.shape)

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

    def backpropagate(self, dLdA):
        if self.f == "relu":
            dLdZ = np.heaviside(self.Z, 0.5) * dLdA
        elif self.f == "softmax":
            dLdZ = self.Aout * (dLdA - np.diagonal(self.Aout.T @ dLdA))
            # dLdZ = self.Aout * dLdA - self.Aout @ (self.Aout.T @ dLdA)  # Change this mf.
        elif self.f == "sigmoid":
            dLdZ = (self.Aout * (1 - self.Aout)) * dLdA # Pointwise, we're okay.
        elif self.f == "none":
            dLdZ = dLdA
        else:  # ignore
            dLdZ = dLdA

        dLdW = np.matmul(self.A, dLdZ.T)  # Figure out this mf too.
        # Exactly what do I want. Currently, A is a stack of column vectors, and dLdZ is a stack of column vectors as well.
        # If we transpose dLdZ, we get a stack of row vectors.
        # Wait, no fucking way. Does this just work????
        # Nice!
        # print(dLdZ.shape)
        dLdW0 = np.sum(dLdZ, axis = 1).reshape((-1,1))
        # print(dLdW0.shape)

        self.t += self.dt

        self.m = self.B1 * self.m + (1 - self.B1) * dLdW
        # print(self.m0.shape)

        self.m0 = self.B1 * self.m0 + (1 - self.B1) * dLdW0
        # print(self.m0.shape)

        self.v = self.B2 * self.v + (1 - self.B2) * dLdW ** 2
        self.v0 = self.B2 * self.v0 + (1 - self.B2) * dLdW0 ** 2

        mhat = self.m / (1 - self.B1 ** self.t)
        mhat0 = self.m0 / (1 - self.B1 ** self.t)

        vhat = self.v / (1 - self.B2 ** self.t)
        vhat0 = self.v0 / (1 - self.B2 ** self.t)

        self.W -= (self.eta / self.t ** 2 / np.sqrt(vhat + self.eps)) * mhat
        self.W0 -= (self.eta / self.t ** 2 / np.sqrt(vhat0 + self.eps)) * mhat0
        # print(self.W0.shape)

        dLdA = np.matmul(self.W, dLdZ)  # Anddddd here. Wait, originally dLdZ was a column vector.
        # Now, it's fleshed out in matrices. Which is like, still okay.

        return dLdA
