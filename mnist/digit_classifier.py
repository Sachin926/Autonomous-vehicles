import tensorflow as tf
from tensorflow import keras
import numpy as np
import cv2 as cv
from tensorflow.keras.models import load_model

(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

#scaling the dataset between 0 to 1 to have better accuracy though it was not requierd in my case as the accuracy already was around .88
cv.imshow("image", x_train[0])
cv.waitKey(0)
x_train = x_train / 255
x_test = x_test / 255

x_train_flat = x_train.reshape(len(x_train), 28*28)
x_test_flat = x_test.reshape(len(x_test), 28*28)

model = keras.Sequential([
			keras.layers.Dense(300, input_shape = (784,), activation = "relu"), 
			keras.layers.Dense(100, activation = "relu"),
			keras.layers.Dense(10, activation = "sigmoid")])	#two hidden layer 
#logging vents on tensor board
tb_callback = tf.keras.callbacks.TensorBoard(log_dir = "logs/", histogram_freq = 1)

model.compile(optimizer = "adam", loss = "sparse_categorical_crossentropy", metrics = ["accuracy"])

model.fit(x_train_flat, y_train, epochs = 5, callbacks = [tb_callback])

model.save("digit_classifier.h5")