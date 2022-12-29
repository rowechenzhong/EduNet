import numpy as np

from Layer import Layer


class ConvolutionBase(Layer):
    def __init__(self, image_size: tuple = None, kernel_size: tuple = None, output_size: tuple = None):
        """

        Creates a Convolution Layer with kernel_size (w, h, d) or output size (w,h,d).

        The input size can be inferred from previous layer.
        Only supply one of kernel_size or output_size needs to be specified; the other will be inferred.

        :param kernel_size:
        :param image_size:
        """

        super().__init__()
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

        self.A = None
        self.Z = None
        self.Aout = None

    def __str__(self):
        return f"Convolution Layer {self.i_size} -> {self.o_size}, Kernel Size {self.k_size}"

    def update_input(self, image_size):
        self.i_size = image_size
        self.image_w, self.image_h, self.image_d = image_size

        if self.k_size is not None:
            self.output_w = (self.image_w - self.kernel_w) + 1
            self.output_h = (self.image_h - self.kernel_h) + 1
            self.output_d = self.image_d + self.kernel_d - 1

            self.o_size = (self.output_w, self.output_h, self.output_d)
        else:  # self.o_size is guaranteed to be not None.
            self.kernel_w = (self.image_w + self.output_w) - 1
            self.kernel_h = (self.image_h + self.output_h) - 1
            self.kernel_d = (self.image_d - self.output_d) + 1

            self.k_size = (self.output_w, self.output_h, self.output_d)