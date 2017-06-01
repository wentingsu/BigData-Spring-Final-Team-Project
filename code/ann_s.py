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

# negative/positive balanced
X_df = pd.read_csv('dataset/X_train_np_balance2.csv')
y_df = pd.read_csv('dataset/y_train_np_balance_label2.csv')

X = X_df.iloc[:, 1:].values
y = y_df.iloc[:, 1:].values

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Part 2 - Build the ANN

# Import the Keras libaries and packages
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout

# Initialising the ANN
classifier = Sequential()

# Adding the input layer and the first hidden layer with dropout
classifier.add(Dense(units = 106, kernel_initializer = 'uniform', activation = 'relu', input_dim = 212))
# classifier.add(Dropout(p = 0.5))

# Adding the second hidden layer
# classifier.add(Dense(units = 106, kernel_initializer = 'uniform', activation = 'relu'))
# classifier.add(Dropout(p = 0.3))

# Adding the third hidden layer
# classifier.add(Dense(units = 106, kernel_initializer = 'uniform', activation = 'relu'))

# Adding the output layer
classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))

# Compiling the ANN
classifier.compile(optimizer = 'rmsprop', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Fiting the ANN to the Training set
classifier.fit(X_train, y_train, batch_size = 32, epochs = 20)

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
plot_model(classifier, to_file='classifier_s.png', show_shapes = True)

'''
# Tuning the ANN
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
from keras.models import Sequential
from keras.layers import Dense
def build_classifier(optimizer):
    classifier = Sequential()
    classifier.add(Dense(units = 106, kernel_initializer = 'uniform', activation = 'relu', input_dim = 212))
    classifier.add(Dense(units = 106, kernel_initializer = 'uniform', activation = 'relu'))
    classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))
    classifier.compile(optimizer = optimizer, loss = 'binary_crossentropy', metrics = ['accuracy'])
    return classifier
classifier = KerasClassifier(build_fn = build_classifier)
parameters = {'batch_size': [25, 32],
              'epochs': [20, 40],
              'optimizer': ['adam', 'rmsprop']}
grid_search = GridSearchCV(estimator = classifier,
                           param_grid = parameters,
                           scoring = 'accuracy',
                           cv = 10)
grid_search = grid_search.fit(X_train, y_train)
best_parameters = grid_search.best_params_
best_accuracy = grid_search.best_score_
print(best_parameters)
print(best_accuracy)
'''