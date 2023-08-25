import numpy as np

from Layer import Layer

from Dense.BGD import Dense

from scipy.special import expit


def softmax(x):
    return np.exp(x - np.max(x)) / np.exp(x - np.max(x)).sum(axis=0)


class Recurrent(Layer):
    def __init__(self, o_size: int, i_size: int = -1, activation: str = "none",
                 eta: float = 0.01, t0: float = 1, dt: float = 0.0001):
        """
        A recurrent layer comes in pairs. The upstream recurrent layer performs
        a dense layer on the data from the previous timestep, then adds it to the
        incoming data. The downstream recurrent layer copies data, and otherwise
        allows the data to pass through unperturbed.

        All parameters are identical to the Dense layer.
        """
        super().__init__()

        self.internal = Dense(o_size, i_size, activation, eta, t0, dt)

    def __str__(self):
        return f"Recurrent Layer {self.i_size[0]} -> {self.o_size[0]}, {self.f} activation"

    def micro(self):
        return f"D{self.f[:2]}{self.o_size[0]}"

    def update_input(self, i_size):
        if type(i_size) == int:
            self.i_size = (i_size, 1)
        elif type(i_size) == tuple:
            self.i_size = i_size

        self.internal.update_input(i_size)

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
