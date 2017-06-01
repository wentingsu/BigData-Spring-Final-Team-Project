#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 13 18:07:14 2017

@author: HuangWei
"""

# Data Preprocessing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

np.set_printoptions(threshold=np.nan)

# Importing the dataset
X_df = pd.read_table(filepath_or_buffer = 'dataset/orange_small_train.data/orange_small_train.data', 
                  sep = '\t').head(1000)
y_df = pd.read_table(filepath_or_buffer = 'dataset/orange_small_train_churn.labels', 
                  sep = '\t', header = None).head(1000)

X_num_df = X_df.ix[:, :190].dropna(axis=1, how = 'all')
X_cat_df = X_df.ix[:, 190:].dropna(axis=1, how = 'all')

X_num = X_num_df.iloc[:, :].values
X_cat = X_cat_df.iloc[:, :].values
y = y_df.iloc[:, :].values

# Processing missing data

from sklearn.preprocessing import Imputer
imputer_num = Imputer(missing_values = 'NaN', strategy = 'median', axis = 0)
imputer_num.fit(X_num)
X_num = imputer_num.transform(X_num)

# Encoding categorical data

cols = X_cat_df.columns

for col in cols:
	X_cat_df[col] = X_cat_df[col].astype('category')
    
X_cat_df[cols] = X_cat_df[cols].apply(lambda x: x.cat.codes)

from sklearn.preprocessing import LabelEncoder
labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)

# Combine numerical and categorical

X_df_combine = pd.DataFrame(X_num).join(X_cat_df)
# X_df_combine = pd.DataFrame(X_num)
X = X_df_combine.iloc[:, :].values

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X = sc_X.fit_transform(X)

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Export to csv
pd.DataFrame(X_train).to_csv('X_train.csv')
pd.DataFrame(y_train).to_csv('y_train_label.csv')
pd.DataFrame(X_test).to_csv('X_test.csv')
pd.DataFrame(y_test).to_csv('y_test_label.csv')

#############################################################

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X_cat[:, :] = labelencoder_X.fit_transform(X_cat[:, :])

onehotencoder = OneHotEncoder(categorical_features = [0])
X = onehotencoder.fit_transform(X).toarray()