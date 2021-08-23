from tensorflow.keras.models import load_model
from tensorflow.keras import datasets
import numpy as np
model = load_model("cifar.h5")
(train_x, train_y), (test_x, test_y) = datasets.cifar10.load_data()
#scaling
test_x = test_x / 255
model.evaluate(test_x, test_y)