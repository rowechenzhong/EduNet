import numpy as np
from scipy.signal import convolve

from ConvolutionBase import ConvolutionBase


class Convolution(ConvolutionBase):
    def __init__(self, image_size: tuple = None, kernel_size: tuple = None, output_size: tuple = None,
                 activation="relu"):
        super().__init__(image_size, kernel_size, output_size)
        self.activation = activation


    def propagate(self, A):
        """

        :param A:
        :return:
        """

        self.A = np.pad(A, ((0, 0), (0, 0), (self.kernel_d - 1, self.kernel_d - 1)), constant_values=0)
        self.Z = convolve(self.A, self.kernels, mode="valid")

        if self.activation == "relu":
            self.Aout = np.maximum(self.Z, 0)
        else:
            self.Aout = self.Z
        return self.Aout

    def backpropagate(self, dLdA):
        """
        Here, dLdA is the amount that we blame each element A in the output.

        Observe that " W * image = A " for some very convoluted definition of " * "

        So we figure out what dLdW was by looking at the image. We update and send dLdA back.

        :param dLdA:
        :return:dLdA
        """
        if self.activation == "relu":
            dLdZ = dLdA * np.heaviside(self.Z, 0.5)
        else:
            dLdZ = dLdA

        dLdW = convolve(np.flip(self.A, (0, 1, 2)), dLdZ, mode='valid')

        self.kernels -= 0.0001 * dLdW  # TODO: self.eta

        dLdApad = np.pad(dLdZ, ((self.kernel_w - 1, self.kernel_w - 1),
                                (self.kernel_h - 1, self.kernel_h - 1),
                                (0, 0))
                         , constant_values=0)

        dLdA = convolve(dLdApad, np.flip(self.kernels, (0, 1, 2)), mode="valid")

        return dLdA
