from tensorflow.keras.models import load_model
from tensorflow.keras import datasets
import numpy as np
import cv2 as cv
(x_train, y_train), (x_test, y_test) = datasets.cifar10.load_data()
#scaling
x_test  = x_test / 255
model = load_model(r"C:\Users\indu\Desktop\DeepLearning\CNN\colab_cnn.h5")

custom = cv.imread("horse2.jpg")
custom = cv.resize(custom, (32, 32))
custom = custom / 255

pred = model.predict_on_batch(custom.reshape(1, 32, 32, 3))

l = ["airplane", "automobile", "bird", "cat", "deer",
				 "dog", "frog", "horse", "ship", "truck"]

def showImage(index):
	image = cv.resize(custom, (128, 128))
	cv.imshow(l[index], mat = image)
	cv.waitKey(0)

print (pred)
print (np.argmax(pred))
showImage(np.argmax(pred))