
# coding: utf-8

# In[1]:

import sys
from pyspark import SparkContext, SparkConf
import pandas as pd
import numpy as np
from pandas import Series, DataFrame
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:

from scipy import stats, integrate


# In[3]:

from sklearn.preprocessing import Imputer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder


# In[5]:

# np.set_printoptions(threshold=np.nan)


# ### Step1: Collect train data a d lable data

# In[4]:

# define the function to get the train data using spark.
def getTrainData(path):
    train_file = sc.textFile(path)
    traindata = train_file.map(lambda line: line.split("\t")).collect()
    X_df = pd.DataFrame(traindata)
    return X_df


# In[5]:

# define the function to get the lable data using spark.
def getLableData(path):
    lable_file = sc.textFile(path)
    labeldata = lable_file.map(lambda line: line.split(" ")).collect()
    y_df = pd.DataFrame(labeldata)
    return y_df


# In[6]:

# get the large size data and lable
X_df = getTrainData('dataset/orange_large_train.data')

# In[7]:

# get the large size data and lable
y_df = getLableData('dataset/orange_large_train_churn.labels.txt')


# ### Step2: Processing Missing Data

# * 2.1 dealing with String Empty Data

# In[9]:

# define the function to process the string empty
def processEmpty(X_df):
    # deal with the string empty 
    X_index = X_df[:1].values.tolist()
    X_df2 = X_df.T.set_index(X_index).T.drop(0)
    X_df3 = X_df2.apply(lambda x: x.str.strip()).replace('', np.nan)
    return X_df3


# In[20]:

X_df3 = processEmpty(X_df)


# * 2.2 dealing with Numerical Missing Data

# In[24]:

# Processing missing data of the numerical columns
def ProcessMissNum(X_df3):
    # extract the numerical colums
    X_num_df = X_df3.ix[:, :14740]
    # drop the columns that all nan
    X_num_df2 = X_num_df.dropna(axis=1, how = 'all')
    # Processing missing numerical data
    X_num = np.array(X_num_df2)
    imputer_num = Imputer(missing_values = 'NaN', strategy = 'median', axis = 0)
    imputer_num.fit(X_num)
    X_num = imputer_num.transform(X_num)
    return X_num


# In[25]:

X_num = ProcessMissNum(X_df3)



# * 2.3 Encoding categorical data

# In[27]:

from ipykernel import kernelapp as app


# In[28]:

# Encoding categorical data
def ProcessMissCat(X_df3):
    # extract the categorical colums
    X_cat_df = X_df3.ix[:,14740:]
    # drop the columns that all nan
    X_cat_df2 = X_cat_df.dropna(axis=1, how = 'all')
    return X_cat_df2

# call the function
X_cat_df2 = ProcessMissCat(X_df3)

# Encoding categorical data
cols = X_cat_df2.columns
for col in cols:
    X_cat_df2[col] = X_cat_df2[col].astype('category')
X_cat_df2[cols] = X_cat_df2[cols].apply(lambda x: x.cat.codes)


# * 2.4 Combine the numerical data and categorical data.

# In[30]:

# Comnime the numerical columns and categorical columns
X_df_combine = pd.concat([pd.DataFrame(X_num), pd.DataFrame(np.array(X_cat_df2))], axis=1)


# In[31]:

X = X_df_combine.iloc[:, :].values


# * 2.5 Lable Encoding

# In[33]:

# deal with the lable y train data 
from sklearn.preprocessing import LabelEncoder


# In[34]:

def encodeLable(y_df):
    y = y_df.iloc[:, :].values
    labelencoder_y = LabelEncoder()
    y_train = labelencoder_y.fit_transform(y)
    return y_train


# In[36]:

y_train = encodeLable(y_df)[:9999]


# ### Step3: Feature Scaling

# In[38]:

# Feature Scaling
from sklearn.preprocessing import StandardScaler


# In[39]:

def featureScal(X):
    sc_X = StandardScaler()
    X_train = sc_X.fit_transform(X)
    return X_train
X_std = featureScal(X)


# ### Step4: PCA

from sklearn.decomposition import PCA


# In[53]:

def DimReduce(X_std, n):
    X = np.array(X_std)
    pca=PCA(n_components=n)
    pca.fit(X)
    X_Train = DataFrame(pca.transform(X))
    return X_Train


# In[54]:

X_Train = DimReduce(X_std, 6000)


# Export to csv
pd.DataFrame(X_Train).to_csv('dataset/pca_large_train.csv')


# In[57]:

# Export to csv
pd.DataFrame(y_train).to_csv('dataset/y_train_large_label.csv')


# In[42]:

# def pcaPlot(X_std):
#     # 1 - Eigendecomposition - Computing Eigenvectors and Eigenvalues
#     # Covariance Matrix
#     mean_vec = np.mean(X_std, axis=0)
#     cov_mat = (X_std - mean_vec).T.dot((X_std - mean_vec)) / (X_std.shape[0]-1)
#     # 2.
#     cov_mat = np.cov(X_std.T)
#     eig_vals, eig_vecs = np.linalg.eig(cov_mat)
#     #3 sort
#     tot = sum(eig_vals)
#     var_exp = [(i / tot)*100 for i in sorted(eig_vals, reverse=True)]
#     cum_var_exp = np.cumsum(var_exp)
#     # plot explained variance ratio
#     return var_exp,cum_var_exp


# In[43]:

# def plotCEV(var_exp,cum_var_exp,n):
#     with plt.style.context('seaborn-whitegrid'):
#         plt.figure(figsize=(6, 4))

#         plt.bar(range(n), var_exp, alpha=0.5, align='center',
#             label='individual explained variance')
#         plt.step(range(n), cum_var_exp, where='mid',
#              label='cumulative explained variance')
#         plt.ylabel('Explained variance ratio')
#         plt.xlabel('Principal components')
#         plt.legend(loc='best')
#         plt.tight_layout()
#         plt.show()


# In[44]:

# var_exp,cum_var_exp = pcaPlot(X_std)

# # In[47]:

# plotCEV(var_exp,cum_var_exp,14890)


# In[50]:

# with plt.style.context('seaborn-whitegrid'):
#     plt.figure(figsize=(6, 4))

#     plt.bar(range(14890), var_exp, alpha=0.5, align='center',
#             label='individual explained variance')
# #     plt.step(range(212), cum_var_exp, where='mid',
# #              label='cumulative explained variance')
#     plt.ylabel('Explained variance ratio')
#     plt.xlabel('Principal components')
#     plt.legend(loc='best')
#     plt.tight_layout()
#     plt.show()


# In[ ]:

# Correlation Matrix


# In[51]:



# In[ ]:

# print(pca.explained_variance_ratio_) 


# ### Large dataset export

# In[56]:




# In[102]:




# In[ ]:




# In[ ]:



