from Model import *

<<<<<<< HEAD
from Convolution import Convolution
from Auxilliary import Flatten
from Dense.Dense import Dense
=======
from Layers.Convolution import Convolution
from Layers.Auxilliary import Flatten, MaxPool
from Layers.Dense import Dense
>>>>>>> master

from keras.datasets import fashion_mnist

if __name__ == "__main__":

    # load data
    (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

    TRAIN_SIZE = len(x_train)
    TEST_SIZE = len(x_test)

    print(f"Training Dataset {TRAIN_SIZE}")
    print(f"Testing Dataset {TEST_SIZE}")

    # normalize inputs from 0-255 to 0-1, reshape to input shape.
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

<<<<<<< HEAD
    BATCH_SIZE = 500
    BATCH_COUNT = TRAIN_SIZE // BATCH_SIZE

    TEST_BATCH_SIZE = TEST_SIZE // 100

    TEST_BATCH_COUNT = TEST_SIZE // TEST_BATCH_SIZE

    print(f"Training: {BATCH_SIZE} points per batch, {BATCH_COUNT} batches per epoch, {TRAIN_SIZE} points total.")
    print(f"Testing: {TEST_BATCH_SIZE} per test, {TEST_SIZE} tests total.")
    for epoch in range(BATCH_COUNT):
        cumulative_loss = 0
        cumulative_correct = 0

        for i in range(epoch * BATCH_SIZE, (epoch + 1) * BATCH_SIZE):
            x = x_train[i]
            y = y_train[i]
            result, failure = Network.cycle(x, y)
            cumulative_loss += failure

            cumulative_correct += CCEcorrect(result, y)
        print(
            f"Epoch = {epoch} Average Loss = {cumulative_loss / BATCH_SIZE},"
            f"Accuracy = {cumulative_correct / BATCH_SIZE}")

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
            f"Testing --- Average Loss = {cumulative_loss / TEST_BATCH_SIZE},"
            f" Accuracy = {cumulative_correct / TEST_BATCH_SIZE}"
        )
=======
    Network.train(x_train, y_train, x_test, y_test, CCEcorrect, batch_size=5000)
>>>>>>> master
