import cv2
import os
import tensorflow as tf
import numpy as np
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, Dense, MaxPooling2D, Flatten
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split

from google.colab import drive
#mount your drive

# Commented out IPython magic to ensure Python compatibility.
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
import random
# cd to your drive
DIRECTORY = 'images'
CATEGORIES = ['car', 'cat', 'dog']
IMG_SIZE = (100, 100)

data = []

for folder in os.listdir(DIRECTORY):
  if(folder == "test"):
    continue
  path = os.path.join(DIRECTORY, folder)
  for img in os.listdir(path):
    try:
      img_path = os.path.join(path, img)
      img_arr = cv2.imread(img_path)
      img_arr = cv2.resize(img_arr, IMG_SIZE)
      img_arr = img_arr/255
      class_num = CATEGORIES.index(folder)
      data.append([img_arr, class_num])
    except Exception as e:
      print(e)

random.shuffle(data)

X = []
y = []

for feature, label in data:
	X.append(feature)
	y.append(label)

X = np.array(X)
y = np.array(y)

import keras
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout
from keras.callbacks import TensorBoard

model = Sequential()

model.add(Conv2D(32, (3,3)))
model.add(MaxPooling2D(pool_size = (2,2)))

model.add(Conv2D(32, (3,3)))
model.add(MaxPooling2D(pool_size = (2,2)))

model.add(Flatten())

model.add(Dense(32, activation = 'relu', input_shape = X[1:]))
model.add(Dense(32, activation = 'relu'))
model.add(Dense(32, activation = 'relu'))
model.add(Dense(3, activation = 'softmax'))

model.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics=['accuracy'])

model.fit(X, y, batch_size = 32, epochs=8, validation_split=0.1)

import urllib
def img2arr(path):
	arr = cv2.imread(path)
	arr = cv2.resize(arr, IMG_SIZE)
	arr = arr/255
	arr = np.array(arr)
	return arr
def url_to_image(url_in):
  with urllib.request.urlopen(url_in) as url:
    arr= np.asarray(bytearray(url.read()))
    arr = cv2.resize(arr, IMG_SIZE)
    arr = arr/255
    arr = np.array(arr)
    return arr


tests=[]
for folder in os.listdir(DIRECTORY):
  if(folder == "test"):
    path = os.path.join(DIRECTORY, folder)
    for img in os.listdir(path):
      try:
        tests.append(img2arr(os.path.join(path, img)))
      except Exception as e:
        print(e)
random.shuffle(tests)
for i in range(10):
  plt.imshow(tests[i])
  P=model.predict(tests[i].reshape(1,100,100,3))
  plt.title(CATEGORIES[P.argmax()])
  plt.show()