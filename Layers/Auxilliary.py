from Core.Layer import Layer
import numpy as np

from Util.Stride import expand3d


class Flatten(Layer):
    def __init__(self, image_size: tuple = None):
        super().__init__()
        self.i_size = None
        if image_size is not None:
            self.update_input(image_size)

    def micro(self):
        return "F"

    def update_input(self, image_size):
        self.i_size = image_size

        image_w, image_h, image_d = image_size

        self.o_size = (image_w * image_h * image_d, 1)

    def __str__(self):
        return f"Flatten Layer {self.i_size} -> {self.o_size[0]}"

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


class MaxPool(Layer):
    def __init__(self, image_size: tuple = None, kernel_size: tuple = (3, 3), stride=2):
        super().__init__()
        self.i_size = None

        self.o_size = None
        self.output_w = self.output_h = self.output_d = None

        self.k_size = kernel_size
        self.kernel_w, self.kernel_h = kernel_size

        self.stride = stride

        # Storage Prop -> Backprop
        self.rows = None
        self.cols = None

        self.image_w, self.image_h, self.image_d = None, None, None
        self.row_offset = self.col_offset = self.depths = None

        if image_size is not None:
            self.update_input(image_size)

    def update_input(self, image_size):
        self.i_size = image_size
        self.image_w, self.image_h, self.image_d = image_size

        # Stride may exit the thing. Fine. Smh.
        self.output_w = (self.image_w - self.kernel_w) // self.stride + 1
        self.output_h = (self.image_h - self.kernel_h) // self.stride + 1
        self.output_d = self.image_d

        self.o_size = (self.output_w, self.output_h, self.output_d)

        self.row_offset = np.tile(self.stride * np.arange(self.output_w).reshape((-1, 1, 1)),
                                  (1, self.output_h, self.output_d))
        self.col_offset = np.tile(self.stride * np.arange(self.output_h).reshape((1, -1, 1)),
                                  (self.output_w, 1, self.output_d))

        self.depths = np.tile(np.arange(self.output_d),
                              (self.output_w, self.output_h, 1))

    def micro(self):
        return "M"

    def __str__(self):
        return f"Maxpool Layer {self.i_size} -> {self.o_size}, Kernel = {self.k_size}, Stride = {self.stride}"

    def test(self, A):
        return np.amax(expand3d(A, self.k_size, self.stride), axis=3)

    def propagate(self, A):
        blow = expand3d(A, self.k_size, self.stride)
        self.rows, self.cols = np.unravel_index(
            np.argmax(blow, axis=3), self.k_size)
        return np.amax(blow, axis=3)

    def backpropagate(self, dLdZ):
        self.rows += self.row_offset
        self.cols += self.col_offset
        dLdA = np.zeros(self.i_size)
        np.add.at(dLdA, (self.rows, self.cols, self.depths), dLdZ)
        return dLdA
