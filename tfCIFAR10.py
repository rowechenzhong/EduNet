import tensorflow as tf

from keras import datasets, layers, models
import matplotlib.pyplot as plt

(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()

# Normalize pixel values to be between 0 and 1
train_images, test_images = train_images / 255.0, test_images / 255.0

class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

#
# plt.figure(figsize=(10,10))
#
# show_start = 50
#
# for i in range(25):
#     plt.subplot(5,5,i+1)
#     plt.xticks([])
#     plt.yticks([])
#     plt.grid(False)
#     plt.imshow(train_images[i+show_start])
#     # The CIFAR labels happen to be arrays,
#     # which is why you need the extra index
#     plt.xlabel(class_names[train_labels[i+show_start][0]])
# plt.show()

# 0.7016 accuracy after 10 epochs. 18s to train each epoch
def triple_convolution():
    model = models.Sequential()
    model.add(layers.Conv2D(10, (3, 3), activation='relu', input_shape=(32, 32, 3)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(40, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)));
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(10, activation='softmax'))
    return model

"""

With dropout, the accuracy can hit -> 0.728 after 13 epochs. I need to check the original...

0.7109
"""


def tcwithDropout():
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.2))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(10, activation='softmax'))
    return model

# .7286

def tcwithDropout2():
    model = models.Sequential()
    model.add(layers.Conv2D(64, (3, 3), activation='relu', input_shape=(32, 32, 3)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(80, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.2))
    model.add(layers.Conv2D(80, (3, 3), activation='relu'))
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='sigmoid'))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(10, activation='softmax'))
    return model


"""
Yay Chonky Model! 4 Convolution layers :0 This is a big boi
Takes around a minute to train on performance XD. Turbo -> um still 1 minute the fans just get louder skull

Epoch 1/10
1563/1563 [==============================] - 61s 39ms/step - loss: 1.4265 - accuracy: 0.4861 - val_loss: 1.2511 - val_accuracy: 0.5652
Epoch 2/10
1563/1563 [==============================] - 60s 38ms/step - loss: 0.9822 - accuracy: 0.6555 - val_loss: 0.9835 - val_accuracy: 0.6489
Epoch 3/10
1563/1563 [==============================] - 60s 38ms/step - loss: 0.8012 - accuracy: 0.7200 - val_loss: 0.8674 - val_accuracy: 0.7007
Epoch 4/10
1563/1563 [==============================] - 60s 38ms/step - loss: 0.6913 - accuracy: 0.7593 - val_loss: 0.8168 - val_accuracy: 0.7168
Epoch 5/10
1563/1563 [==============================] - 60s 39ms/step - loss: 0.5945 - accuracy: 0.7932 - val_loss: 0.8086 - val_accuracy: 0.7272
Epoch 6/10
1563/1563 [==============================] - 58s 37ms/step - loss: 0.5189 - accuracy: 0.8177 - val_loss: 0.8401 - val_accuracy: 0.7146
Epoch 7/10
1563/1563 [==============================] - 58s 37ms/step - loss: 0.4484 - accuracy: 0.8427 - val_loss: 0.8351 - val_accuracy: 0.7334
Epoch 8/10
1563/1563 [==============================] - 58s 37ms/step - loss: 0.3886 - accuracy: 0.8622 - val_loss: 0.8362 - val_accuracy: 0.7414 ********** Peak
Epoch 9/10
1563/1563 [==============================] - 58s 37ms/step - loss: 0.3265 - accuracy: 0.8842 - val_loss: 0.9851 - val_accuracy: 0.7142
Epoch 10/10
1563/1563 [==============================] - 58s 37ms/step - loss: 0.2885 - accuracy: 0.8986 - val_loss: 1.0714 - val_accuracy: 0.7183


"""

def quad_convolution():
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3), padding="same"))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(128, (3, 3), activation='relu', padding="same"))
    model.add(layers.MaxPooling2D((1, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu', padding="same"))
    model.add(layers.MaxPooling2D((2, 1)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu', padding="same"))
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(10, activation='softmax'))
    return model


# More Dense
# 0.7112, 18s training time, after 10 epochs.
# Try 100 epochs? No 0.700 is about it.

def triple_convolution_more_dense():
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(30, activation='relu'))
    model.add(layers.Dense(10, activation='softmax'))
    return model


# This is completely garbage, you need convolution layers apparently.

def extradense():
    model = models.Sequential()
    model.add(layers.Flatten(input_shape=(32, 32, 3)))
    model.add(layers.Dense(800, activation='relu'))
    model.add(layers.Dense(400, activation='relu'))
    model.add(layers.Dense(200, activation='relu'))
    model.add(layers.Dense(100, activation='relu'))
    model.add(layers.Dense(50, activation='relu'))
    model.add(layers.Dense(10))
    return model

if __name__ == "__main__":
    for construct in (triple_convolution,):


        model = construct()
        model.summary()

        model.compile(optimizer='adam',
                      loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                      #from_logits = True will take in elements of R, False is probability distro, also default.
                      metrics=['accuracy'])

        history = model.fit(train_images, train_labels, epochs=40,
                            validation_data=(test_images, test_labels))

        plt.plot(history.history['accuracy'], label='accuracy')
        plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.ylim([0.5, 1])
        plt.legend(loc='lower right')
        plt.show()

        test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)

