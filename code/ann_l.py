#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 15 13:22:34 2017

@author: HuangWei
"""

# Part 1 - Data Preprocessing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
'''
X_train_df = pd.read_csv('X_train.csv')
X_test_df = pd.read_csv('X_test.csv')
y_train_df = pd.read_csv('y_train_label.csv')
y_test_df = pd.read_csv('y_test_label.csv')


X_train = X_train_df.iloc[:, 1:].values
X_test = X_test_df.iloc[:, 1:].values
y_train = y_train_df.iloc[:, 1:].values
y_test = y_test_df.iloc[:, 1:].values
'''

X_df = pd.read_csv('dataset/pca_large_train.csv')
y_df = pd.read_csv('dataset/y_train_large_label.csv')

X = X_df.iloc[:, 1:].values
y = y_df.iloc[1:, 1:].values

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.35, random_state = 0)

# Part 2 - Build the ANN

# Import the Keras libaries and packages
import keras
from keras.models import Sequential
from keras.layers import Dense

# Initialising the ANN
classifier = Sequential()

# Adding the input layer and the first hidden layer
classifier.add(Dense(units = 3000, kernel_initializer = 'uniform', activation = 'relu', input_dim = 6000))

# Adding the output layer
classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))

# Compiling the ANN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Fiting the ANN to the Training set
classifier.fit(X_train, y_train, batch_size = 10, epochs = 10)

# Part 3 - Making predictions and evaluating the model

# Fitting classifier to the Training set
# Create your classifier here

# Predicting the Test set results
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
score = classifier.evaluate(X_test, y_test, verbose = 0)

print("Confusion Matrix: ")
print(cm)
print("Loss: ", score[0])
print("Test accuracy: ", score[1])

from keras.utils import plot_model
plot_model(classifier, to_file='classifier_l.png', show_shapes = True)

# Predicting a single new observation
# new_prediction = classifier.predict(sc.transform(np.array([[0.0, 0, 600, 1, 40, 3, 600000, 2, 1, 1, 50000]])))
# new_prediction = (new_prediction > 0.5)