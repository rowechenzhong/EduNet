class Flatten:
    def __init__(self, image_size: tuple = None):
        self.i_size = None
        if image_size is not None:
            self.update_input(image_size)

    def update_input(self, image_size):
        self.i_size = image_size

        self.image_w, self.image_h, self.image_d = image_size

        self.o_size = (self.image_w * self.image_h * self.image_d, 1)



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
