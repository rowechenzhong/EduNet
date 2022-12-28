from Model import *

from Dense import Dense

from keras.datasets import fashion_mnist

if __name__ == "__main__":

    # load data
    (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

    TRAIN_SIZE = len(x_train)
    TEST_SIZE = len(x_test)

    print(f"Training Dataset {TRAIN_SIZE}")
    print(f"Testing Dataset {TEST_SIZE}")

    # normalize inputs from 0-255 to 0-1
    x_train = x_train / 255
    x_test = x_test / 255

    x_train = x_train.reshape((x_train.shape[0], 784, 1)).astype('float32')
    x_test = x_test.reshape((x_test.shape[0], 784, 1)).astype('float32')

    Network = Model(CCEloss, CCEdLdA)
    Network.join(Dense(i_size=784, o_size=700, activation="sigmoid"))
    Network.join(Dense(o_size=300, activation="sigmmoid"))
    Network.join(Dense(o_size=10, activation="softmax"))

    Network.compile()

    print(Network)
    print(Network.micro())

    BATCH_SIZE = TRAIN_SIZE // 100
    BATCH_COUNT = TRAIN_SIZE // BATCH_SIZE

    TEST_BATCH_SIZE = TEST_SIZE

    TEST_BATCH_COUNT = TEST_SIZE // TEST_BATCH_SIZE

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
            f"Epoch = {epoch} ({BATCH_SIZE} per batch) Average Loss = {cumulative_loss / BATCH_SIZE}, Accuracy = {cumulative_correct / BATCH_SIZE}")

        cumulative_loss = 0
        cumulative_correct = 0

        which_test_batch = epoch % TEST_BATCH_COUNT

        for i in range(which_test_batch * TEST_BATCH_SIZE, (which_test_batch + 1) * TEST_BATCH_SIZE):
            x = x_test[i]
            y = y_test[i]
            result, failure = Network.test(x, y)
            cumulative_loss += failure
            cumulative_correct += CCEcorrect(result, y)

        print(
            f"Testing ({TEST_BATCH_SIZE} per test) --- Average Loss = {cumulative_loss / TEST_BATCH_SIZE}, Accuracy = {cumulative_correct / TEST_BATCH_SIZE}")
    Network.save()