from keras.datasets import fashion_mnist
from Layers.Dense.BGD import Dense
from Core.Loss import BatchCategoricalCrossEntropy
from Core.Model import *

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

    x_train = x_train.reshape((TRAIN_SIZE, 784)).astype('float32')
    x_test = x_test.reshape((TEST_SIZE, 784, 1)).astype('float32')

    Network = Model(BatchCategoricalCrossEntropy)
    Network.join(Dense(i_size=784, o_size=700, activation="sigmoid"))
    Network.join(Dense(o_size=300, activation="sigmmoid"))
    Network.join(Dense(o_size=10, activation="softmax"))

    Network.compile()

    print(Network)
    print(Network.micro())

    BATCH_SIZE = 1000
    BATCH_COUNT = TRAIN_SIZE // BATCH_SIZE

    x_test = x_test.reshape((TEST_SIZE, 784)).T

    print(
        f"Training: {BATCH_SIZE} points per batch, {BATCH_COUNT} batches per epoch, {TRAIN_SIZE} points total.")
    print(f"Testing: {TEST_SIZE} per test, {TEST_SIZE} tests total.")

    for epoch in range(20):
        cumulative_loss = 0
        cumulative_correct = 0
        for batch in range(BATCH_COUNT):
            # Lol sure.
            x = x_train[batch * BATCH_SIZE: (batch + 1)
                        * BATCH_SIZE].reshape((BATCH_SIZE, 784)).T
            # Maybe? Idk lol
            y = y_train[batch * BATCH_SIZE: (batch + 1) * BATCH_SIZE]

            result, failure, other = Network.cycle(x, y)
            cumulative_loss += failure.sum()

            cumulative_correct += other[0]
        print(f"Epoch = {epoch} "
              f"Average Loss = {cumulative_loss / TRAIN_SIZE}, "
              f"Accuracy = {cumulative_correct / TRAIN_SIZE}")
        # Print shapes
        # print(
        #     f"Epoch = {epoch} "
        #     f"Average Loss = {cumulative_loss.shape}, "
        #     f"Accuracy = {cumulative_correct.shape}")

        cumulative_loss = 0
        cumulative_correct = 0

        result, failure, other = Network.test(x_test, y_test)
        cumulative_loss += failure.sum()
        cumulative_correct += other[0]
        # debug: print all shapes.
        # print(
        #     f"Cumulative Loss is {cumulative_loss.shape}, Cumulative Correct is {cumulative_correct.shape}")
        print(
            f"Testing ({TEST_SIZE} per test) --- Average Loss = {cumulative_loss / TEST_SIZE}, "
            f"Accuracy = {cumulative_correct / TEST_SIZE}")
    Network.save()
