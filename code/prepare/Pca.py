
# coding: utf-8

# In[2]:

import sys
from pyspark import SparkContext, SparkConf
import pandas as pd
import numpy as np
from pandas import Series, DataFrame


# In[3]:

from sklearn.preprocessing import Imputer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder


# In[44]:

df = pd.read_csv('framenew.csv')


# In[29]:

df = df.dropna(axis=1, how = 'all')


# In[47]:

X_num_df = df.ix[:, :190]


# In[50]:

X_cat_df = df.ix[:,190:]


# In[51]:

X_num_df2 = X_num_df.dropna(axis=1, how = 'all')


# In[52]:

X_cat_df2 = X_cat_df.dropna(axis=1, how = 'all')


# In[68]:

X_num = np.array(X_num_df2)


# In[69]:

imputer_num = Imputer(missing_values = 'NaN', strategy = 'median', axis = 0)


# In[70]:

imputer_num.fit(X_num)


# In[71]:

X_num = imputer_num.transform(X_num)


# In[53]:

from ipykernel import kernelapp as app


# In[54]:

cols = X_cat_df2.columns


# In[55]:

for col in cols:
    X_cat_df2[col] = X_cat_df2[col].astype('category')
X_cat_df2[cols] = X_cat_df2[cols].apply(lambda x: x.cat.codes)


# In[73]:

X_df_combine = pd.concat([DataFrame(X_num),X_cat_df2], axis=1)


# In[74]:

X_df_combine.to_csv('data.csv')


# In[8]:

y_df = pd.read_csv('y_train_label.csv')


# In[ ]:




# In[4]:

y_df = pd.read_csv('data.csv')


# In[5]:

X_data = y_df.ix[:,1:]


# In[6]:

X = X_data.values


# In[9]:

y=y_df.ix[:,1:].values


# In[10]:

from sklearn.preprocessing import StandardScaler
X_std = StandardScaler().fit_transform(X)


# In[17]:

df = DataFrame(X_std)


# In[20]:

import numpy as np
mean_vec = np.mean(X_std, axis=0)
cov_mat = (X_std - mean_vec).T.dot((X_std - mean_vec)) / (X_std.shape[0]-1)
print('Covariance matrix \n%s' %cov_mat)


# In[13]:

print('NumPy covariance matrix: \n%s' %np.cov(X_std.T))


# In[39]:

cov_mat = np.cov(X_std.T)
eig_vals, eig_vecs = np.linalg.eig(cov_mat)

# print('Eigenvectors \n%s' %eig_vecs)
# print('\nEigenvalues \n%s' %eig_vals)


# In[44]:

eig_pairs = [(np.abs(eig_vals[i]), eig_vecs[:,i]) for i in range(len(eig_vals))]
eig_pairs.sort(reverse=True)


# In[46]:

DataFrame(eig_pairs).head()


# In[51]:

eig_pairs = [(np.abs(eig_vals[i]), eig_vecs[:,i]) for i in range(len(eig_vals))]

# Sort the (eigenvalue, eigenvector) tuples from high to low
eig_pairs.sort(key=lambda x: x[0], reverse=True)

# Visually confirm that the list is correctly sorted by decreasing eigenvalues
print('Eigenvalues in descending order:')
for i in eig_pairs:
    print(i[0])


# In[102]:

DataFrame(eig_pairs[:100]).T


# In[107]:

feature=eig_pairs[:10]


# In[108]:

feature


# In[110]:

for one in feature:
    new_data_reduced=np.transpose(np.dot(one,np.transpose(X_std)))


# In[64]:

DataFrame(new_data_reduced).head()


# In[128]:

tot = sum(eig_vals)
var_exp = [(i / tot)*100 for i in sorted(eig_vals, reverse=True)]
cum_var_exp = np.cumsum(var_exp)


# In[ ]:




# In[129]:

import matplotlib.pyplot as plt


# In[130]:

with plt.style.context('seaborn-whitegrid'):
    plt.figure(figsize=(6, 3))

    plt.bar(range(212), var_exp, alpha=0.5, align='center',
            label='individual explained variance')
    plt.step(range(212), cum_var_exp, where='mid',
             label='cumulative explained variance')
    plt.ylabel('Explained variance ratio')
    plt.xlabel('Principal components')
    plt.legend(loc='best')
    plt.tight_layout()
    plt.show()


# In[28]:

matrix_w = np.hstack((eig_pairs[0][1].reshape(212,1),
                      eig_pairs[1][1].reshape(212,1)))

print('Matrix W:\n', matrix_w)


# In[29]:

Y = X_std.dot(matrix_w)


# In[41]:

with plt.style.context('seaborn-whitegrid'):
    plt.figure(figsize=(6, 4))
    for lab, col in zip(('Iris-setosa', 'Iris-versicolor'),
                        ('blue', 'red')):
        plt.scatter(Y[y==lab, 0],
                    Y[y==lab, 1],
                    label=lab,
                    c=col)
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.legend(loc='lower center')
    plt.tight_layout()
    plt.show()


# In[67]:

from sklearn.decomposition import PCA


# In[111]:

X = np.array(X_std)


# In[124]:

pca=PCA(n_components=1)


# In[125]:

pca.fit(X)


# In[126]:

DataFrame(pca.transform(X)).head()


# In[96]:

DataFrame(pca.transform(X)).to_csv('eg1.csv')


# In[ ]:



