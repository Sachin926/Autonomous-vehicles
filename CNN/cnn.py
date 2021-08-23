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
test_x = test_x / 255;


model = keras.Sequential([
			#CNN just like dense we can have any number of CNN layers
			layers.Conv2D(filters = 32, kernel_size = (3, 3), activation = "relu", input_shape = (32, 32, 3)),
			layers.MaxPooling2D((2, 2)),
			layers.Conv2D(filters = 32, kernel_size = (3, 3), activation = "relu", input_shape = (32, 32, 3)),
			layers.MaxPooling2D((2, 2)),
			#typical ANN layers
			layers.Flatten(input_shape = (32, 32, 3)), 
			keras.layers.Dense(64, activation = "relu"),
			keras.layers.Dense(10, activation = "sigmoid")])
model.compile(optimizer= "adam", 
	loss = "sparse_categorical_crossentropy", metrics = ["accuracy"])

model.fit(train_x, train_y, epochs = 10)
model.save(r"C:\Users\indu\Desktop\DeepLearning\CNN\cnn.h5")