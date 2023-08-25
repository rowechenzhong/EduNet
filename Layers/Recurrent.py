import numpy as np
from Dense.BGD import Dense
from Core.Layer import Layer
from Util.AdamArray import *


def softmax(x):
    return np.exp(x - np.max(x)) / np.exp(x - np.max(x)).sum(axis=0)


"""
A recurrent layer comes in pairs. The upstream recurrent layer performs
a dense layer on the data from the previous timestep, then adds it to the
incoming data. The downstream recurrent layer copies data, and otherwise
allows the data to pass through unperturbed.

All parameters are identical to the Dense layer.
"""


class DownStreamRecurrent:
    def __init__(self, o_size: int):
        super().__init__()
        self.o_size = (o_size, 1)
        self.copy = np.zeros((o_size, 1))

    def __str__(self):
        return f"Downstream Recurrent Layer {self.o_size[0]} activation"

    def micro(self):
        return f"DR{self.o_size[0]}"

    def propagate(self, A) -> np.ndarray:
        self.copy = A
        return A

    def backpropagate(self, dLdA) -> np.ndarray:
        return dLdA


class UpstreamRecurrent(Layer):
    def __init__(self, twin: DownStreamRecurrent, o_size: int, past_size: int, activation: str = "none"):
        super().__init__()

        self.internal = Dense(o_size, past_size, activation)
        self.twin = twin

    def __str__(self):
        return f"Upstream Recurrent Layer injects {self.internal.i_size[0]} {self.internal.f} activation"

    def micro(self):
        return f"UR{self.internal.f[:2]}{self.internal.o_size[0]}"

    def propagate(self, A) -> np.ndarray:
        return A + self.internal.propagate(self.twin.copy)

    def backpropagate(self, dLdA) -> np.ndarray:
        self.internal.backpropagate(dLdA)

        return dLdA
