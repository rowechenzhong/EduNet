# https://machinelearningmastery.com/handwritten-digit-recognition-using-convolutional-neural-networks-python-keras/

# Baseline MLP for MNIST dataset

import tensorflow as tf
# from tensorflow.python import keras
from keras.datasets import fashion_mnist
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D
from keras.utils import to_categorical

import matplotlib.pyplot as plt




# load data
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()


#

# plot 4 images as gray scale
plt.subplot(221)
plt.imshow(x_train[0], cmap=plt.get_cmap('gray'))
plt.subplot(222)
plt.imshow(x_train[1], cmap=plt.get_cmap('gray'))
plt.subplot(223)
plt.imshow(x_train[2], cmap=plt.get_cmap('gray'))
plt.subplot(224)
plt.imshow(x_train[3], cmap=plt.get_cmap('gray'))
# show the plot
plt.show()


# normalize inputs from 0-255 to 0-1
x_train = x_train / 255
x_test = x_test / 255

# ---------- 1d data



# flatten 28*28 images to a 784 vector for each image
num_pixels = x_train.shape[1] * x_train.shape[2]
x_train1d = x_train.reshape((x_train.shape[0], num_pixels)).astype('float32')
x_test1d = x_test.reshape((x_test.shape[0], num_pixels)).astype('float32')
# one hot encode outputs
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
num_classes = y_test.shape[1]

# ----------- 2d Data
x_train = x_train[..., tf.newaxis].astype("float32")
x_test = x_test[..., tf.newaxis].astype("float32")


# from_tensor_slices works like zip(), and batch() makes... batches, stupid.
# shuffle does something that I don't really care about
train_ds = tf.data.Dataset.from_tensor_slices(
    (x_train, y_train)).shuffle(10000).batch(32)

test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(32)



# 0.9847 Accuracy

# define baseline model
def baseline_model():
	# create model
	model = Sequential()

	model.add(Dense(num_pixels, input_shape=(num_pixels,), kernel_initializer='normal', activation='relu'))
	model.add(Dense(num_classes, kernel_initializer='normal', activation='softmax'))
	# Compile model
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	return model



	# 0.9277 Accuracy

def zero_layers_model():
	# create model
	model = Sequential()

	model.add(Dense(num_classes, input_shape=(num_pixels,), kernel_initializer='normal', activation='softmax'))
	# model.add(Dense(num_classes, kernel_initializer='normal', activation='softmax'))
	# Compile model
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	return model



	# 0.9870

def two_layers_model():
	# create model
	model = Sequential()

	# 784 -> 784 -> 200 -> 10.

	model.add(Dense(num_pixels, input_shape=(num_pixels,), kernel_initializer='normal', activation='relu'))
	# model.add(Dense(200, kernel_initializer='normal', activation='relu'))
	model.add(Dense(num_classes, kernel_initializer='normal', activation='softmax'))
	# Compile model
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	return model


	# 784 -> 784 -> 50 -> 50 -> 50 -> 10.
	# 0.9852

def Lots_of_layers_model():
	# create model
	model = Sequential()


	model.add(Dense(num_pixels, input_shape=(num_pixels,), kernel_initializer='normal', activation='relu'))
	model.add(Dense(50, kernel_initializer='normal', activation='relu'))
	model.add(Dense(50, kernel_initializer='normal', activation='relu'))
	model.add(Dense(50, kernel_initializer='normal', activation='relu'))
	model.add(Dense(num_classes, kernel_initializer='normal', activation='softmax'))
	# Compile model
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	return model




	# 0.9863

def Pyramid():
	# create model
	model = Sequential()


	model.add(Dense(600, input_shape=(num_pixels,), kernel_initializer='normal', activation='relu'))
	model.add(Dense(300, kernel_initializer='normal', activation='relu'))
	model.add(Dense(100, kernel_initializer='normal', activation='relu'))
	model.add(Dense(50, kernel_initializer='normal', activation='relu'))
	model.add(Dense(num_classes, kernel_initializer='normal', activation='softmax'))
	# Compile model
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	return model



def Convolutions():
	# create model
	model = Sequential()

	model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
	model.add(MaxPooling2D((2, 2)))

	model.add(Dense(600, input_shape=(num_pixels,), kernel_initializer='normal', activation='relu'))
	model.add(Dense(300, kernel_initializer='normal', activation='relu'))
	model.add(Dense(100, kernel_initializer='normal', activation='relu'))
	model.add(Dense(50, kernel_initializer='normal', activation='relu'))
	model.add(Dense(num_classes, kernel_initializer='normal', activation='softmax'))
	# Compile model
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	return model


# build the model
for construct in (baseline_model,):
	model = construct()

	# Fit the model
	model.fit(x_train1d, y_train, validation_data=(x_test1d, y_test), epochs=50, batch_size=1000, verbose=2)
	# Final evaluation of the model
	# scores = model.evaluate(X_test, y_test, verbose=0)
	# print("Baseline Error: %.2f%%" % (100-scores[1]*100))