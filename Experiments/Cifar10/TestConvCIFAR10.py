from Layers.Dense.Dense import Dense
from Core.Loss import CategoricalCrossEntropy
from Core.Model import *

from Layers.Convolution.Convolution import Convolution
from Layers.Auxilliary import Flatten, MaxPool

from keras.datasets import cifar10

import matplotlib.pyplot as plt


if __name__ == "__main__":

    (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                   'dog', 'frog', 'horse', 'ship', 'truck']

    plt.figure(figsize=(10, 10))

    show_start = 50

    for i in range(25):
        plt.subplot(5, 5, i+1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(x_train[i+show_start])
        # The CIFAR labels happen to be arrays,
        # which is why you need the extra index
        plt.xlabel(class_names[y_train[i+show_start][0]])
    plt.show()

    TRAIN_SIZE = len(x_train)
    TEST_SIZE = len(x_test)

    print(f"Training Dataset {TRAIN_SIZE}")
    print(f"Testing Dataset {TEST_SIZE}")

    # normalize inputs from 0-255 to 0-1, reshape to input size.
    x_train = x_train / 255
    x_test = x_test / 255

    x_train = x_train.astype('float32')
    y_train = y_train.reshape((y_train.size,))
    x_test = x_test.astype('float32')
    y_test = y_test.reshape((y_test.size,))

    Network = Model(CategoricalCrossEntropy)
    Network.join(Convolution((32, 32, 3), (3, 3, 10)))
    Network.join(Convolution(kernel_size=(3, 3, 10)))
    Network.join(Convolution(kernel_size=(3, 3, 10)))
    Network.join(Flatten())
    Network.join(Dense(10, activation="softmax"))

    Network.compile()

    print(Network)
    print(Network.micro())

    Network.train(x_train, y_train, x_test, y_test)
    Network.save()
