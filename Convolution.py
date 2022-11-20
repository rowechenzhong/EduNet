import numpy as np


class Flatten():
    def __init__(self, image_size):

        self.i_size = image_size

        self.image_w, self.image_h, self.image_d = image_size

    def propagate(self, A):
        # Z = np.zeros((self.image_w * self.image_h * self.image_d, 1))
        #
        # for i in range(self.image_w):
        #     for j in range(self.image_h):
        #         for k in range(self.image_d):
        #             Z[self.image_w * self.image_h * k + self.image_w * j + i][0] = A[i][j][k]
        # return Z
        return A.reshape((-1, 1))

    def backpropagate(self, dLdZ):
        # dLdA = np.zeros(self.i_size)
        #
        # for i in range(self.image_w):
        #     for j in range(self.image_h):
        #         for k in range(self.image_d):
        #             dLdA[i][j][k] = dLdZ[self.image_w * self.image_h * k + self.image_w * j + i]
        # return dLdA
        return dLdZ.reshape(self.i_size)


class Convolution():
    def __init__(self, kernel_size, image_size):
        """
        
        kernel_num * kernel_size * kernel_size

        :param kernel_num:
        :param kernel_size:
        """

        self.k_size = kernel_size

        self.kernel_w, self.kernel_h, self.kernel_d = kernel_size

        # self.stride = stride


        self.i_size = image_size

        self.image_w, self.image_h, self.image_d = image_size

        self.output_w = (self.image_w - self.kernel_w) + 1  # (self.image_w - self.f) / self.stride + 1
        self.output_h = (self.image_h - self.kernel_h) + 1  # (self.image_h - self.f) / self.stride + 1
        self.output_d = self.image_d + self.kernel_d - 1

        self.o_size = (self.output_w, self.output_h, self.output_d)

        self.kernels = np.random.randn(*self.k_size) / (self.kernel_d ** 2)

        self.A = None

    # def patches_generator(self, image):
    #     self.image = image
    #
    #     for h in range(self.output_h):
    #         for w in range(self.output_w):
    #             patch = image[h : (h+self.kernel_size), w : (w+ self.kernel_size)]
    #             yield patch, h, w

    def propagate(self, A):
        """

        :param image: h, w, [depth]
        :return:
        """

        Z = np.zeros(self.o_size)

        self.A = np.pad(A, ((0, 0), (0, 0), (self.kernel_d - 1, self.kernel_d - 1)), constant_values=0)
        #
        # print(f"input A {self.i_size}")
        # print(f"Padded A {self.A.shape}")
        # print(f"output Z {self.o_size}")
        # print(f"kernel {(self.kernel_size, self.kernel_size, self.kernel_num)} == {self.kernels.shape}")

        for i in range(self.output_w):
            for j in range(self.output_h):
                for k in range(self.output_d):
                    # print(f"A[box] {self.A[i:i+self.kernel_size,j:j+self.kernel_size,k:k+self.kernel_num].shape}")

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


        for i in range(self.output_w):
            for j in range(self.output_h):
                for k in range(self.output_d):
                    dLdW += self.A[
                            i:i + self.kernel_w,
                            j:j + self.kernel_h,
                            k:k + self.kernel_d
                            ] * dLdZ[i][j][k]


        self.kernels -= 0.0001 * dLdW  # TODO: self.eta

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


        dLdA = np.zeros(self.i_size)

        for i in range(self.image_w):
            for j in range(self.image_h):
                for k in range(self.image_d):
                    # print(f"dLdZPad[box] {dLdZPad[i:i + self.kernel_size, j:j + self.kernel_size, k:k + self.kernel_num].shape}")
                    dLdA[i][j][k] = np.tensordot(dLdZPad[
                                                  i:i + self.kernel_w,
                                                  j:j + self.kernel_h,
                                                  k:k + self.kernel_d
                                                  ], np.flip(self.kernels, (0, 1, 2)), 3)

        return dLdA