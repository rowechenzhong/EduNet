import numpy as np

def softmax(x):
    return np.exp(x - np.max(x)) / np.exp(x - np.max(x)).sum()

class Dense():
    def __init__(self, m: int, n: int, f: str, eta: float, t0: float, dt: float):
        self.m = m
        self.n = n
        self.f = f

        self.eta = eta

        self.t = t0
        self.dt = dt

        self.W = np.random.normal(0, 1/m, (m, n))
        self.W0 = np.random.normal(0, 1, (n,1) )

        self.m = np.zeros(self.W.shape)
        self.m0 = np.zeros(self.W0.shape)
        self.v = np.zeros(self.W.shape)
        self.v0 = np.zeros(self.W0.shape)

        self.B1 = 0.9
        self.B2 = 0.999
        self.eps = 10**-8


        # m x n numpy array. Guassian init.

        self.Z = None
        self.A = None
        self.Aout = None

    def propagate(self, A):
        self.A = A # Store input for backprop

        self.Z = np.matmul(self.W.T, self.A)

        self.Z+= self.W0


        if self.f == "relu":
            self.Aout = np.maximum(0, self.Z)
        elif self.f == "softmax":
            self.Aout = softmax(self.Z)
        else:
            print("Yr bad")
            self.Aout = self.Z

        return self.Aout

    def backpropagate(self, dLdA):

        if self.f == "relu":
            dAdZ = np.diagflat(np.heaviside(self.Z, 0.5))

        elif self.f == "softmax":
            SM = self.Aout.reshape((-1, 1))
            dAdZ = np.diagflat(self.Aout) - np.dot(SM, SM.T)

        else: # ignore
            dAdZ = np.diagflat(np.ones((self.n,)))

        dLdZ = np.matmul(dAdZ, dLdA)

        dLdW = np.matmul(self.A, dLdZ.T)

        dLdW0 = dLdZ

        self.t += self.dt

        self.m = self.B1 * self.m + (1 - self.B1) * dLdW
        self.m0 = self.B1 * self.m0 + (1 - self.B1) * dLdW0

        self.v = self.B2 * self.v + (1 - self.B2) * dLdW**2
        self.v0 = self.B2 * self.v0 + (1 - self.B2) * dLdW0**2

        mhat = self.m / (1 - self.B1 ** self.t)
        mhat0 = self.m0 / (1 - self.B1 ** self.t)

        vhat = self.v / (1 - self.B2 ** self.t)
        vhat0 = self.v0 / (1 - self.B2 ** self.t)

        self.W -= (self.eta / self.t ** 2 / np.sqrt(vhat + self.eps) ) * mhat
        self.W0 -= (self.eta / self.t ** 2 / np.sqrt(vhat0 + self.eps) ) * mhat0

        dLdA = np.matmul(self.W, dLdZ)

        return dLdA

