
# coding: utf-8

# In[254]:

import sys
from pyspark import SparkContext, SparkConf
import pandas as pd
import numpy as np
from pandas import Series, DataFrame
import matplotlib.pyplot as plt
import seaborn as sns

# In[107]:

from scipy import stats, integrate

# In[2]:

from sklearn.preprocessing import Imputer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

# In[5]:

# np.set_printoptions(threshold=np.nan)

# ### Step1: Collect train data a d lable data

# In[13]:

# define the function to get the train data using spark.
def getTrainData(path):
    train_file = sc.textFile(path)
    traindata = train_file.map(lambda line: line.split("\t")).collect()
    X_df = pd.DataFrame(traindata)
    return X_df

# In[16]:

# define the function to get the lable data using spark.
def getLableData(path):
    lable_file = sc.textFile(path)
    labeldata = lable_file.map(lambda line: line.split(" ")).collect()
    y_df = pd.DataFrame(labeldata)
    return y_df

# In[14]:

# get the small size data and lable
X_df = getTrainData('dataset/orange_small_train.data')
y_df = getLableData('dataset/orange_small_train_churn.labels.txt')

# In[12]:

# ### Step2: Processing Missing Data
# * 2.1 dealing with String Empty Data

# In[34]:

# define the function to process the string empty
def processEmpty(X_df):
    # deal with the string empty 
    X_index = X_df[:1].values.tolist()
    X_df2 = X_df.T.set_index(X_index).T.drop(0)
    X_df3 = X_df2.apply(lambda x: x.str.strip()).replace('', np.nan)
    return X_df3

# In[35]:

X_df3 = processEmpty(X_df)

# In[37]:

# * 2.2 dealing with Numerical Missing Data

# In[38]:

# Processing missing data of the numerical columns
def ProcessMissNum(X_df3):
    # extract the numerical colums
    X_num_df = X_df3.ix[:, :190]
    # drop the columns that all nan
    X_num_df2 = X_num_df.dropna(axis=1, how = 'all')
    # Processing missing numerical data
    X_num = np.array(X_num_df2)
    imputer_num = Imputer(missing_values = 'NaN', strategy = 'median', axis = 0)
    imputer_num.fit(X_num)
    X_num = imputer_num.transform(X_num)
    return X_num


# In[39]:

X_num = ProcessMissNum(X_df3)



# * 2.3 Encoding categorical data

# In[64]:

from ipykernel import kernelapp as app

# In[58]:

# Encoding categorical data
def ProcessMissCat(X_df3):
    # extract the categorical colums
    X_cat_df = X_df3.ix[:,190:]
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

# In[60]:

# Comnime the numerical columns and categorical columns
X_df_combine = pd.concat([pd.DataFrame(X_num), pd.DataFrame(np.array(X_cat_df2))], axis=1)

# In[70]:

X = X_df_combine.iloc[:, :].values


# In[61]:

# * 2.5 Lable Encoding

# In[66]:

# deal with the lable y train data 
from sklearn.preprocessing import LabelEncoder

# In[67]:

def encodeLable(y_df):
    y = y_df.iloc[:, :].values
    labelencoder_y = LabelEncoder()
    y_train = labelencoder_y.fit_transform(y)
    return y_train


# In[258]:

y_train = encodeLable(y_df)


# ### Step3: Feature Scaling

# In[74]:

# Feature Scaling
from sklearn.preprocessing import StandardScaler

# In[86]:

def featureScal(X):
    sc_X = StandardScaler()
    X_train = sc_X.fit_transform(X)
    return X_train
X_std = featureScal(X)


# In[240]:
# In[35]:


# ### Step4: PCA

from sklearn.decomposition import PCA

# In[262]:

def DimReduce(X_std, n):
    X = np.array(X_std)
    pca=PCA(n_components=n)
    pca.fit(X)
    X_Train = DataFrame(pca.transform(X))
    return X_Train


# In[263]:

X_Train = DimReduce(X_std, 150)

# Export to csv
pd.DataFrame(X_std).to_csv('dataset/X_train.csv')

# In[ ]:

# Export to csv
pd.DataFrame(y_train).to_csv('dataset/y_train_label.csv')


# In[97]:

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


# In[101]:

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

# In[98]:

# var_exp,cum_var_exp = pcaPlot(X_std)


# # In[100]:

# plotCEV(var_exp,cum_var_exp,212)


# In[95]:

# with plt.style.context('seaborn-whitegrid'):
#     plt.figure(figsize=(6, 4))

#     plt.bar(range(212), var_exp, alpha=0.5, align='center',
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


# In[89]:


# In[ ]:

# print(pca.explained_variance_ratio_) 


# ### Small dataset export

# In[ ]:


# In[102]:




# In[ ]:




# In[ ]:



