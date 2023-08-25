from keras.datasets import fashion_mnist
from Layers.Dense.Dense import Dense
from Core.Loss import CategoricalCrossEntropy
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

    x_train = x_train.reshape((TRAIN_SIZE, 784, 1)).astype('float32')
    x_test = x_test.reshape((TEST_SIZE, 784, 1)).astype('float32')

    Network = Model(CategoricalCrossEntropy)
    Network.join(Dense(i_size=784, o_size=700, activation="sigmoid"))
    Network.join(Dense(o_size=300, activation="sigmmoid"))
    Network.join(Dense(o_size=10, activation="softmax"))

    Network.compile()

    print(Network)
    print(Network.micro())

    Network.train(x_train, y_train, x_test, y_test,
                  batch_size=TRAIN_SIZE // 100)
    Network.save()
