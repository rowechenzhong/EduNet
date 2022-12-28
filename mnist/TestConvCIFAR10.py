from Model import *

from Convolution import Convolution
from Auxilliary import Flatten
from Dense import Dense

from keras.datasets import cifar10

import matplotlib.pyplot as plt


if __name__ == "__main__":

    (x_train, y_train), (x_test, y_test) = cifar10.load_data()


    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                   'dog', 'frog', 'horse', 'ship', 'truck']


    plt.figure(figsize=(10,10))

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


    #
    # # load data
    # (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
    #

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

    Network = Model(CCEloss, CCEdLdA)
    Network.join(Convolution((32, 32, 3), (3, 3, 10)))
    Network.join(Convolution(kernel_size=(3, 3, 10)))
    Network.join(Convolution(kernel_size=(3, 3, 10)))
    Network.join(Flatten())
    Network.join(Dense(10, activation="softmax"))

    Network.compile()

    print(Network)
    print(Network.micro())

    BATCH_SIZE = TRAIN_SIZE
    BATCH_COUNT = TRAIN_SIZE // BATCH_SIZE

    TEST_BATCH_SIZE = TEST_SIZE

    TEST_BATCH_COUNT = TEST_SIZE // TEST_BATCH_SIZE

    for epoch in range(10 * BATCH_COUNT):
        cumulative_loss = 0
        cumulative_correct = 0


        which_train_batch = epoch % BATCH_COUNT

        for i in range(which_train_batch * BATCH_SIZE, (which_train_batch + 1) * BATCH_SIZE):
            x = x_train[i]
            y = y_train[i]
            result, failure = Network.cycle(x, y)
            cumulative_loss += failure

            cumulative_correct += CCEcorrect(result, y)
        print(
            f"Epoch = {epoch} ({BATCH_SIZE} per batch) Average Loss = {cumulative_loss / BATCH_SIZE}, Accuracy = {cumulative_correct / BATCH_SIZE}")

        cumulative_loss = 0
        cumulative_correct = 0

        which_test_batch = epoch % TEST_BATCH_COUNT

        for i in range(which_test_batch * TEST_BATCH_SIZE, (which_test_batch + 1) * TEST_BATCH_SIZE):
            x = x_test[i]
            y = y_test[i]
            result, failure = Network.test(x,  y)
            cumulative_loss += failure
            cumulative_correct += CCEcorrect(result, y)

        print(
            f"Testing --- Average Loss = {cumulative_loss / TEST_BATCH_SIZE}, Accuracy = {cumulative_correct / TEST_BATCH_SIZE}")
