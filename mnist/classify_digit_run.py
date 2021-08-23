from tensorflow.keras.models import load_model
from tensorflow import keras
import mnist_fromat_resize as img
import numpy as np
import cv2 as cv

(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
x_train = x_train / 255
x_test = x_test / 255

img.small = img.small / 255		

x_train_flat = x_train.reshape(len(x_train), 28*28)
x_test_flat = x_test.reshape(len(x_test), 28*28)

model = load_model("digit_classifier.h5")

print ((img.small.shape))
pred = model.predict_on_batch(x_test_flat[19].reshape(1, 784))

print (pred)
print (np.argmax(pred))
cv.imshow("image", mat = x_test[19])
cv.waitKey(0)


pred = model.predict_on_batch(img.small.reshape(1, 784))
print (pred)
print (np.argmax(pred))
cv.imshow("test", mat = img.small)
cv.waitKey(0)