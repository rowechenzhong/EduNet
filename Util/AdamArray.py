import numpy as np

class AdamArray():
    def __init__(self, shape: tuple, eta: float = 0.01, t0: float = 1, dt: float = 0.0001):
        """
        Array that can be updated via Adam.

        Initializes with a normal distribution with SD = 1 / shape[0]

        #TODO: Transpose the stupid weight matrix please

        :param shape:
        """
        self.shape = shape
        self.W = np.random.normal(0, 1 / shape[0], shape)
        self.m = np.zeros(shape)
        self.v = np.zeros(shape)

        self.t = t0
        self.dt = dt

        self.eta = eta
        self.B1 = 0.9
        self.B2 = 0.999
        self.eps = 10 ** -8


    def update(self, dLdW):
        self.t += self.dt
        self.m = self.B1 * self.m + (1 - self.B1) * dLdW
        self.v = self.B2 * self.v + (1 - self.B2) * dLdW ** 2
        mhat = self.m / (1 - self.B1 ** self.t)
        vhat = self.v / (1 - self.B2 ** self.t)
        self.W -= (self.eta / self.t ** 2 / np.sqrt(vhat + self.eps)) * mhat
