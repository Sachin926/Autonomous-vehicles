import tensorflow as tf
from tensorflow.keras import datasets, layers, models
from tensorflow import keras
import numpy as np
import cv2 as cv
from tensorflow.keras.models import load_model
(train_x, train_y), (test_x, test_y) = datasets.cifar10.load_data()

l = ["airplane", "automobile", "bird", "cat", "deer",
				 "dog", "frog", "horse", "ship", "truck"]

def showImage(index):
	cv.imshow(l[test_y[index][0]], mat = test_x[index])
	cv.waitKey(0)


#scaling
train_x = train_x / 255;
test_y = test_y / 255;


model = keras.Sequential([
			layers.Flatten(input_shape = (32, 32, 3)), 
			keras.layers.Dense(3000, activation = "relu"),
			keras.layers.Dense(1000, activation = "relu"),
			keras.layers.Dense(10, activation = "sigmoid")])
model.compile(optimizer= "SGD", 
	loss = "sparse_categorical_crossentropy", metrics = ["accuracy"])

model.fit(train_x, train_y, epochs = 10)
model.save("cifar.h5")