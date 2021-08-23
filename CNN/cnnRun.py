from tensorflow.keras.models import load_model
from tensorflow.keras import datasets
import numpy as np
import cv2 as cv
(x_train, y_train), (x_test, y_test) = datasets.cifar10.load_data()
#scaling
x_test  = x_test / 255
model = load_model(r"C:\Users\indu\Desktop\DeepLearning\CNN\colab_cnn.h5")
#model.evaluate(x_test, y_test)
pred = model.predict(x_test[74].reshape(1, 32, 32, 3))

l = ["airplane", "automobile", "bird", "cat", "deer",
				 "dog", "frog", "horse", "ship", "truck"]

def showImage(index):
	image = cv.resize(x_test[index], (128, 128))
	cv.imshow(l[y_test[index][0]], mat = image)
	cv.waitKey(0)

print (pred)
showImage(np.argmax(pred))