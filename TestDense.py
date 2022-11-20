import matplotlib.pyplot as plt

from Model import *

from Dense import Dense

from keras.datasets import fashion_mnist

if __name__ == "__main__":

    # load data
    (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

    TRAIN_SIZE = len(x_train)

    print(TRAIN_SIZE)

    TEST_SIZE = len(x_test)

    print(TEST_SIZE)

    # plot 4 images as gray scale
    plt.subplot(241)
    plt.imshow(x_train[0], cmap=plt.get_cmap('gray'))
    plt.subplot(242)
    plt.imshow(x_train[1], cmap=plt.get_cmap('gray'))
    plt.subplot(243)
    plt.imshow(x_train[2], cmap=plt.get_cmap('gray'))
    plt.subplot(244)
    plt.imshow(x_train[3], cmap=plt.get_cmap('gray'))

    plt.subplot(245)
    plt.imshow(x_train[4], cmap=plt.get_cmap('gray'))
    plt.subplot(246)
    plt.imshow(x_train[5], cmap=plt.get_cmap('gray'))
    plt.subplot(247)
    plt.imshow(x_train[6], cmap=plt.get_cmap('gray'))
    plt.subplot(248)
    plt.imshow(x_train[7], cmap=plt.get_cmap('gray'))

    # show the plot
    plt.show()

    # normalize inputs from 0-255 to 0-1
    x_train = x_train / 255
    x_test = x_test / 255

    # flatten 28*28 images to a 784 vector for each image
    num_pixels = x_train.shape[1] * x_train.shape[2]
    x_train1d = x_train.reshape((x_train.shape[0], num_pixels)).astype('float32')
    x_test1d = x_test.reshape((x_test.shape[0], num_pixels)).astype('float32')
    # # one hot encode outputs
    # y_train = to_categorical(y_train)
    # y_test = to_categorical(y_test)
    # num_classes = y_test.shape[1]

    Network = (Dense(784, 784, "relu", 0.01, 1, 0.001), Dense(784, 10, "softmax", 0.01, 1, 0.001))

    BATCH_COUNT = 100
    BATCH_SIZE = TRAIN_SIZE // BATCH_COUNT
    # hello

    for epoch in range(1000):
        # Train!

        which_batch = epoch % BATCH_COUNT

        cumulative_loss = 0
        cumulative_correct = 0

        for i in range(which_batch * BATCH_SIZE, (which_batch + 1) * BATCH_SIZE):
            x = x_train1d[i]
            y = y_train[i]
            result = feed_forward(Network, x.reshape(784, 1))
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

        for i in range(TEST_SIZE):
            x = x_test1d[i]
            y = y_test[i]
            result = feed_forward(Network, x.reshape(784, 1))
            failure = loss(result, y)
            dLdA = get_dLdA(result, y)
            cumulative_loss += failure
            cumulative_correct += correct(result, y)

        print(f"Testing --- Average Loss = {cumulative_loss / TEST_SIZE}, Accuracy = {cumulative_correct / TEST_SIZE}")

