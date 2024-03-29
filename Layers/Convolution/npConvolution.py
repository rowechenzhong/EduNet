import numpy as np

from ConvolutionBase import ConvolutionBase


class Convolution(ConvolutionBase):
    def __init__(self, image_size: tuple = None, kernel_size: tuple = None, output_size: tuple = None, eta: float = 0.0001):
        super().__init__(image_size, kernel_size, output_size)

        self.kernels = np.random.randn(*self.k_size) / (self.kernel_d ** 2)
        self.eta = eta

    def propagate(self, A):
        """

        :param image: h, w, [depth]
        :return:
        """

        Z = np.zeros(self.o_size)

        self.A = np.pad(A, ((0, 0), (0, 0), (self.kernel_d - 1,
                        self.kernel_d - 1)), constant_values=0)

        for i in range(self.output_w):
            for j in range(self.output_h):
                for k in range(self.output_d):

                    Z[i][j][k] = np.tensordot(self.A[
                                              i:i + self.kernel_w,
                                              j:j + self.kernel_h,
                                              k:k + self.kernel_d
                                              ], self.kernels, 3)

        return Z

    def backpropagate(self, dLdZ):
        """
        Here, dLdA is the amount that we blame each element A in the output.

        Observe that " W * image = A " for some very convoluted definition of " * "

        So we figure out what dLdW was by looking at the image. We update and send dLdW back.

        :param dLdA:
        :return:
        """

        dLdW = np.zeros(self.kernels.shape)

        for i in range(self.output_w):
            for j in range(self.output_h):
                for k in range(self.output_d):
                    dLdW += self.A[
                        i:i + self.kernel_w,
                        j:j + self.kernel_h,
                        k:k + self.kernel_d
                    ] * dLdZ[i][j][k]

        self.kernels -= self.eta * dLdW

        dLdZPad = np.pad(dLdZ, ((self.kernel_w - 1, self.kernel_w - 1),
                                (self.kernel_h - 1, self.kernel_h - 1),
                                (0, 0)), constant_values=0)

        dLdA = np.zeros(self.i_size)

        for i in range(self.image_w):
            for j in range(self.image_h):
                for k in range(self.image_d):
                    dLdA[i][j][k] = np.tensordot(dLdZPad[
                        i:i + self.kernel_w,
                        j:j + self.kernel_h,
                        k:k + self.kernel_d
                    ], np.flip(self.kernels, (0, 1, 2)), 3)

        return dLdA
