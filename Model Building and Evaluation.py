#This python notebook will guide you to build a model and evaluate it.
#Real Time Face Mask Detection Model
#First we have import all the required libraries for our project
import os
import random
import numpy as np
import cv2
from keras.preprocessing import image
from keras.applications.vgg16 import VGG16
from keras import Sequential
from keras.layers import Dense, Flatten
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense
from sklearn.metrics import classification_report, confusion_matrix


# Assuming the dataset is downloaded and available in the following local directories
categories = ['with_mask', 'without_mask']
data = []
for category in categories:
    path = os.path.join('Real Time face mask detection\\data', category)
    label = categories.index(category)
    
    for file in os.listdir(path):
        img_path = os.path.join(path, file)
        img = cv2.imread(img_path)
        img = cv2.resize(img, (224, 224))
        data.append([img, label])


# Randomly shuffle the data
random.shuffle(data)
X = []
y = []
for features, label in data:
    X.append(features)
    y.append(label)

X = np.array(X)
y = np.array(y)
X = X / 255.0

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Load VGG16 model with pre-trained weights
vgg = VGG16(include_top=False, input_shape=(224, 224, 3), weights='imagenet')
model = Sequential()
model.add(vgg)
model.add(Flatten())
model.add(Dense(1, activation='sigmoid'))

# Fine-tune some of the VGG layers
for layer in vgg.layers[:-4]:  # Freeze all but the last 4 layers
    layer.trainable = False

model.compile(optimizer='Adam', loss='binary_crossentropy', metrics=['accuracy'])

# Assuming X_train, y_train, X_test, and y_test are defined
model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))

# Save the model
model.save('facemaskdetectionmodel.h5')

# Evaluate the model
y_pred = model.predict(X_test)
y_pred = (y_pred > 0.5).astype(int)  # Convert probabilities to binary labels

# Print classification report
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Print confusion matrix
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
