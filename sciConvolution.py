import numpy as np
from scipy.signal import convolve



class Convolution():
    def __init__(self, image_size: tuple = None, kernel_size: tuple = None, output_size: tuple = None):
        """

        Creates a Convolution Layer with kernel_size (w, h, d) or output size (w,h,d).

        The input size can be inferred from previous layer.
        Only supply one of kernel_size or output_size needs to be specified; the other will be inferred.

        :param kernel_size:
        :param image_size:
        """

        self.i_size = None
        self.image_w = self.image_h = self.image_d = None
        self.o_size = None
        self.output_w = self.output_h = self.output_d = None
        self.k_size = None
        self.kernel_w = self.kernel_h = self.kernel_d = None

        if kernel_size is None:
            if output_size is None:
                raise ValueError("Either a Kernel Size or an Output Size must be specified")
            else:
                self.o_size = output_size
                self.output_w, self.output_h, self.output_d = output_size
        else:
            self.k_size = kernel_size
            self.kernel_w, self.kernel_h, self.kernel_d = kernel_size

        if image_size is not None:
            self.update_input(image_size)

        self.kernels = np.random.randn(*self.k_size) / (self.kernel_d ** 2)

        self.A = None

    def update_input(self, image_size):

        self.i_size = image_size
        self.image_w, self.image_h, self.image_d = image_size

        if self.k_size is not None:
            self.output_w = (self.image_w - self.kernel_w) + 1
            self.output_h = (self.image_h - self.kernel_h) + 1
            self.output_d = self.image_d + self.kernel_d - 1

            self.o_size = (self.output_w, self.output_h, self.output_d)
        else: #self.o_size is guaranteed to be not None.
            self.kernel_w = (self.image_w + self.output_w) - 1
            self.kernel_h = (self.image_h + self.output_h) - 1
            self.kernel_d = (self.image_d - self.output_d) + 1

            self.k_size = (self.output_w, self.output_h, self.output_d)

    def propagate(self, A):
        """

        :param image: h, w, [depth]
        :return:
        """

        self.A = np.pad(A, ((0, 0), (0, 0), (self.kernel_d - 1, self.kernel_d - 1)), constant_values=0)
        Z = convolve(self.A, self.kernels, mode = "valid")
        # print(f"input A {self.i_size}")
        # print(f"Padded A {self.A.shape}")
        # print(f"output Z {self.o_size}")
        # print(f"kernel {(self.kernel_size, self.kernel_size, self.kernel_num)} == {self.kernels.shape}")


        return Z

    def backpropagate(self, dLdZ):
        """
        Here, dLdA is the amount that we blame each element A in the output.

        Observe that " W * image = A " for some very convoluted definition of " * "

        So we figure out what dLdW was by looking at the image. We update and send dLdW back.

        :param dLdA:
        :return:
        """

        # dLdW = np.zeros(self.kernels.shape)
        #
        # print("WHat")
        # print(f"dLdW {dLdW.shape}")
        # print(f"dLdZ {dLdZ.shape}")
        #
        # print(f"padded A {self.A.shape}")
        #
        # print(f"output {self.o_size}")
        # print(f"input {self.i_size}")
        # print(f"kernel {(self.kernel_size, self.kernel_size, self.kernel_num)}")

        # for i in range(self.output_w):
        #     for j in range(self.output_h):
        #         for k in range(self.output_d):
        #             dLdW += self.A[
        #                     i:i + self.kernel_w,
        #                     j:j + self.kernel_h,
        #                     k:k + self.kernel_d
        #                     ] * dLdZ[i][j][k]

        # dLdW = np.zeros(self.kernels.shape)
        # for i in range(self.kernel_w):
        #     for j in range(self.kernel_h):
        #         for k in range(self.kernel_d):
        #             dLdW[i][j][k] = np.tensordot(self.A[i: i+self.output_w,
        #                                  j:j+self.output_h,
        #                                  k:k+self.output_d], dLdZ, 3)

        dLdW = convolve(np.flip(self.A, (0,1,2)), dLdZ, mode = 'valid')
        # print(dLdW2 - dLdW)

        self.kernels -= 0.0001 * dLdW #np.flip(dLdW, (0, 1, 2))  # TODO: self.eta

        #
        # print(f"padded A {self.A.shape}")
        #
        # print(f"output {self.o_size}")
        # print(f"input {self.i_size}")
        # print(f"kernel {(self.kernel_size, self.kernel_size, self.kernel_num)}")

        dLdZPad = np.pad(dLdZ, ((self.kernel_w - 1, self.kernel_w - 1),
                                (self.kernel_h - 1, self.kernel_h - 1),
                                (0, 0))
                         , constant_values=0)

        # print(f"dLdZPad {dLdZPad.shape}")

        dLdA = convolve(dLdZPad, np.flip(self.kernels, (0, 1, 2)), mode = "valid")

        return dLdA
