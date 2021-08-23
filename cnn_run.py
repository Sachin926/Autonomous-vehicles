from tensorflow.keras.models import load_model
from tensorflow.keras import datasets
model = load_model(r"C:\Users\indu\Desktop\DeepLearning\CNN\cnn.h5")
(train_x, train_y), (test_x, test_y) = datasets.cifar10.load_data()

l = ["airplane", "automobile", "bird", "cat", "deer",
				 "dog", "frog", "horse", "ship", "truck"]

#scaling
train_x = train_x / 255;
test_x = test_x / 255;

def showImage(index):
	cv.imshow(l[test_y[index][0]], mat = test_x[index])
	cv.waitKey(0)
model.evaluate(test_x, test_y)