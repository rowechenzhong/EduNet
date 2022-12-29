from Model import *

from Layers.Convolution import Convolution
from Layers.Auxilliary import Flatten, MaxPool
from Layers.Dense import Dense

from keras.datasets import fashion_mnist

if __name__ == "__main__":

    # load data
    (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

    TRAIN_SIZE = len(x_train)
    TEST_SIZE = len(x_test)

    print(f"Training Dataset {TRAIN_SIZE}")
    print(f"Testing Dataset {TEST_SIZE}")

    # normalize inputs from 0-255 to 0-1, reshape to input size.
    x_train = x_train / 255
    x_test = x_test / 255

    x_train = x_train.reshape((*x_train.shape, 1)).astype('float32')
    x_test = x_test.reshape((*x_test.shape, 1)).astype('float32')

    # Network = Model(CCEloss, CCEdLdA)
    # Network.join(Convolution((28, 28, 1), (10, 10, 3)))
    # Network.join(Convolution(kernel_size=(10, 10, 3)))
    # Network.join(Flatten())
    # Network.join(Dense(10, activation="softmax"))
    #
    # Network.compile()

    Network = Model(CCEloss, CCEdLdA)
    Network.join(Convolution((28, 28, 1), (4, 4, 10)))
    Network.join(MaxPool(kernel_size=(4, 4), stride=3))
    Network.join(Convolution(kernel_size=(3, 3, 10)))
    Network.join(Flatten())
    Network.join(Dense(10, activation="softmax"))
    Network.compile()

    print(Network)

    Network.train(x_train, y_train, x_test, y_test, CCEcorrect, batch_size=5000)