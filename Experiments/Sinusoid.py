import numpy as np
from Model import *

from Layers.Dense import Dense

if __name__ == "__main__":

    TRAIN_SIZE = 10000
    TEST_SIZE = 1000

    # load data
    # x_train = np.arange(TRAIN_SIZE).reshape((-1, 1, 1)).astype('float32') / (TRAIN_SIZE / 10)
    # y_train = np.sin(x_train)
    # LOL OKAY doing data that is *not* scrambled is very bad.

    x_train = np.random.uniform(0, 10, (TRAIN_SIZE, 1, 1))
    y_train = np.sin(x_train)


    x_test = np.random.uniform(0, 10, (TEST_SIZE, 1, 1))
    y_test = np.sin(x_test)

    print(f"Training Dataset {TRAIN_SIZE}")
    print(f"Testing Dataset {TEST_SIZE}")

    Network = Model(sqloss, sqdLdA)
    Network.join(Dense(i_size=1, o_size=10, activation="relu", eta=0.001))
    Network.join(Dense(o_size=1, eta=0.001))

    Network.compile()

    print(Network)
    print(Network.micro())

    BATCH_SIZE = TRAIN_SIZE
    BATCH_COUNT = TRAIN_SIZE // BATCH_SIZE

    TEST_BATCH_SIZE = TEST_SIZE

    TEST_BATCH_COUNT = TEST_SIZE // TEST_BATCH_SIZE

    for epoch in range(100):
        cumulative_loss = 0
        cumulative_correct = 0

        train_batch = epoch % BATCH_COUNT

        for i in range(train_batch * BATCH_SIZE, (train_batch + 1) * BATCH_SIZE):
            x = x_train[i]
            y = y_train[i]
            result, failure = Network.cycle(x, y)
            cumulative_loss += failure
        print(
            f"Epoch = {epoch} ({BATCH_SIZE} per batch) Average Loss = {cumulative_loss / BATCH_SIZE}")

        cumulative_loss = 0
        cumulative_correct = 0

        test_batch = epoch % TEST_BATCH_COUNT

        for i in range(test_batch * TEST_BATCH_SIZE, (test_batch + 1) * TEST_BATCH_SIZE):
            x = x_test[i]
            y = y_test[i]
            result, failure = Network.test(x, y)
            cumulative_loss += failure

        print(" "*60 + f"Testing ({TEST_BATCH_SIZE} per test) --- Average Loss = {cumulative_loss / TEST_BATCH_SIZE}")
    Network.save()
