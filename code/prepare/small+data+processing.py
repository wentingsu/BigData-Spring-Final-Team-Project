
# coding: utf-8

# In[1]:

import sys
from pyspark import SparkContext, SparkConf
import pandas as pd
import numpy as np
from pandas import Series, DataFrame


# In[2]:

from sklearn.preprocessing import Imputer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder


# In[3]:

np.set_printoptions(threshold=np.nan)


# In[3]:

test_file = sc.textFile("/Users/wentingsu/Downloads/orange_small_train.data/orange_small_train.data")


# In[75]:

lable_file = sc.textFile("/Users/wentingsu/Downloads/orange_small_train_churn.labels.txt")


# In[91]:

smalldata = test_file.map(lambda line: line.split("\t")).collect()


# In[76]:

labeldata = lable_file.map(lambda line: line.split(" ")).collect()


# In[103]:

X_df = pd.DataFrame(smalldata)


# In[78]:

y_df = pd.DataFrame(labeldata)


# In[59]:

# framenew.to_csv('framenew.csv')


# In[104]:

X_index = X_df[:1].values.tolist()


# In[105]:

X_df2 = X_df.T.set_index(X_index).T.drop(0)


# In[106]:

X_df3 = X_df2.apply(lambda x: x.str.strip()).replace('', np.nan)


# In[110]:

# extract the numerical colums
X_num_df = X_df3.ix[:, :190]


# In[111]:

# extract the categorical colums
X_cat_df = X_df3.ix[:,190:]


# In[112]:

# drop the columns that all nan
X_num_df2 = X_num_df.dropna(axis=1, how = 'all')


# In[113]:

X_cat_df2 = X_cat_df.dropna(axis=1, how = 'all')


# In[114]:

# Processing missing numerical data
X_num = np.array(X_num_df2)


# In[115]:

imputer_num = Imputer(missing_values = 'NaN', strategy = 'median', axis = 0)


# In[116]:

imputer_num.fit(X_num)


# In[117]:

X_num = imputer_num.transform(X_num)


# In[ ]:

# Encoding categorical data


# In[119]:

from ipykernel import kernelapp as app


# In[136]:

cols = X_cat_df2.columns


# In[137]:

for col in cols:
    X_cat_df2[col] = X_cat_df2[col].astype('category')
X_cat_df2[cols] = X_cat_df2[cols].apply(lambda x: x.cat.codes)


# In[139]:

X_df_combine = pd.concat([pd.DataFrame(X_num), pd.DataFrame(np.array(X_cat_df2))], axis=1)


# In[140]:

X_df_combine.head()


# In[141]:

X = X_df_combine.iloc[:, :].values


# In[142]:

# Feature Scaling
from sklearn.preprocessing import StandardScaler


# In[143]:

sc_X = StandardScaler()


# In[144]:

X_train = sc_X.fit_transform(X)


# In[82]:

# Export to csv
pd.DataFrame(X_train).to_csv('X_train.csv')


# In[146]:

# deal with the lable y train data 
from sklearn.preprocessing import LabelEncoder


# In[147]:

y = y_df.iloc[:, :].values


# In[150]:

labelencoder_y = LabelEncoder()


# In[151]:

y_train = labelencoder_y.fit_transform(y)


# In[152]:

# Export to csv
pd.DataFrame(y_train).to_csv('y_train_label.csv')

