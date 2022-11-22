import matplotlib.pyplot as plt

from Model import *

from sciConvolution import Convolution
from Flatten import Flatten
from Dense import Dense

from keras.datasets import fashion_mnist

if __name__ == "__main__":

    # load data
    (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

    TRAIN_SIZE = len(x_train)
    TEST_SIZE = len(x_test)

    print(f"Training Dataset {TRAIN_SIZE}")

    print(f"Testing Dataset {TEST_SIZE}")


    # plot 4 images as gray scale
    # plt.subplot(241)
    # plt.imshow(x_train[0], cmap=plt.get_cmap('gray'))
    # plt.subplot(242)
    # plt.imshow(x_train[1], cmap=plt.get_cmap('gray'))
    # plt.subplot(243)
    # plt.imshow(x_train[2], cmap=plt.get_cmap('gray'))
    # plt.subplot(244)
    # plt.imshow(x_train[3], cmap=plt.get_cmap('gray'))
    #
    # plt.subplot(245)
    # plt.imshow(x_train[4], cmap=plt.get_cmap('gray'))
    # plt.subplot(246)
    # plt.imshow(x_train[5], cmap=plt.get_cmap('gray'))
    # plt.subplot(247)
    # plt.imshow(x_train[6], cmap=plt.get_cmap('gray'))
    # plt.subplot(248)
    # plt.imshow(x_train[7], cmap=plt.get_cmap('gray'))
    #
    # # show the plot
    # plt.show()

    # normalize inputs from 0-255 to 0-1
    x_train = x_train / 255
    x_test = x_test / 255

    # flatten 28*28 images to a 784 vector for each image
    # num_pixels = x_train.shape[1] * x_train.shape[2]
    x_train3d = x_train.reshape((*x_train.shape, 1)).astype('float32')

    # print(x_train3d.shape)
    # print(x_train3d[0])

    x_test3d = x_test.reshape((*x_test.shape, 1)).astype('float32')
    # # one hot encode outputs
    # y_train = to_categorical(y_train)
    # y_test = to_categorical(y_test)
    # num_classes = y_test.shape[1]

    Network = (Convolution((28, 28, 1), (10, 10, 3)),
               Convolution((19, 19, 3), (10, 10, 3)),
               Flatten((10, 10, 5)),
               Dense(500, 10, "softmax", 0.01, 1, 0.001))

    BATCH_SIZE = 100
    BATCH_COUNT = TRAIN_SIZE // BATCH_SIZE

    TEST_BATCH_SIZE = 500

    TEST_BATCH_COUNT = TEST_SIZE // TEST_BATCH_SIZE

    for epoch in range(1000):
        # Train!

        which_batch = epoch % BATCH_COUNT

        cumulative_loss = 0
        cumulative_correct = 0

        for i in range(which_batch * BATCH_SIZE, (which_batch + 1) * BATCH_SIZE):
            x = x_train3d[i]
            y = y_train[i]
            result = feed_forward(Network, x)
            failure = loss(result, y)
            dLdA = get_dLdA(result, y)
            back_forward(Network, dLdA)
            cumulative_loss += failure

            cumulative_correct += correct(result, y)

        # for i in range(3):
        #     random_test = random.randint(0, TRAIN_SIZE - 1)
        #     x = x_train1d[random_test]
        #     y = y_train[random_test]
        #     result = feed_forward(Network, x.reshape(784, 1))
        #     print(result)

        print(
            f"Epoch = {epoch} ({BATCH_SIZE} per batch) Average Loss = {cumulative_loss / BATCH_SIZE}, Accuracy = {cumulative_correct / BATCH_SIZE}")

        cumulative_loss = 0
        cumulative_correct = 0

        which_test_batch = epoch % TEST_BATCH_COUNT

        for i in range(which_test_batch*TEST_BATCH_SIZE, (which_test_batch + 1) * TEST_BATCH_SIZE):
            x = x_test3d[i]
            y = y_test[i]
            result = feed_forward(Network, x)
            failure = loss(result, y)
            dLdA = get_dLdA(result, y)
            cumulative_loss += failure
            cumulative_correct += correct(result, y)

        print(f"Testing --- Average Loss = {cumulative_loss / TEST_BATCH_SIZE}, Accuracy = {cumulative_correct / TEST_BATCH_SIZE}")

