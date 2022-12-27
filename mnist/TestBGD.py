from Model import *

from Dense.BGD import Dense

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

    Network = Model(CCEBatchloss, CCEBatchdLdA)
    Network.join(Dense(i_size=784, o_size=700, activation="sigmoid", eta = 0.001))
    Network.join(Dense(o_size=300, activation="sigmmoid", eta = 0.001))
    Network.join(Dense(o_size=10, activation="softmax", eta = 0.001))

    Network.compile()

    print(Network)
    print(Network.micro())

    BATCH_SIZE = 1000
    BATCH_COUNT = TRAIN_SIZE // BATCH_SIZE

    TEST_BATCH_SIZE = TEST_SIZE

    TEST_BATCH_COUNT = TEST_SIZE // TEST_BATCH_SIZE

    x_test = x_test.reshape((TEST_SIZE, 784)).T

    for epoch in range(20):
        cumulative_loss = 0
        cumulative_correct = 0
        for batch in range(BATCH_COUNT):
            x = x_train[batch * BATCH_SIZE: (batch + 1) * BATCH_SIZE].reshape((BATCH_SIZE, 784)).T # Lol sure.
            y = y_train[batch * BATCH_SIZE: (batch + 1) * BATCH_SIZE] # Maybe? Idk lol

            result, failure = Network.cycle(x, y)
            cumulative_loss += failure

            cumulative_correct += CCEBatchcorrect(result, y)
        print(f"Epoch = {batch} ({BATCH_SIZE} per batch) Average Loss = {cumulative_loss / TRAIN_SIZE}, Accuracy = {cumulative_correct / TRAIN_SIZE}")

        cumulative_loss = 0
        cumulative_correct = 0

        t_batch = batch % TEST_BATCH_COUNT

        # x = x_test.reshape((TEST_SIZE, 784))
        # y = y_test
        # x = x_test[t_batch * TEST_BATCH_SIZE: (batch + 1) * TEST_BATCH_SIZE].reshape((TEST_BATCH_SIZE, 784)).T # Lol sure.
        # y = y_test[t_batch * TEST_BATCH_SIZE: (batch + 1) * TEST_BATCH_SIZE] # Maybe? Idk lol

        result, failure = Network.test(x_test, y_test)
        cumulative_loss += failure
        cumulative_correct += CCEBatchcorrect(result, y_test)

        print(
            f"Testing ({TEST_BATCH_SIZE} per test) --- Average Loss = {cumulative_loss / TEST_BATCH_SIZE}, Accuracy = {cumulative_correct / TEST_BATCH_SIZE}")
    Network.save()