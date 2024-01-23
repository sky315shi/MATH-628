#!/usr/bin/env python
# coding: utf-8

# ## MATH 628 FINAL PROJECT
# ### Chunlin Shi   Noah Collins

# In[1]:


import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
data = pd.read_excel('data.xlsx')
data


# In[2]:


data_ret = data[['Ticker Symbol','Names Date','Returns without Dividends']]
data_ret


# In[3]:


data_ret = data_ret.pivot_table(index='Names Date', columns='Ticker Symbol', values='Returns without Dividends')
data_ret


# In[4]:


data_ret.info()


# In[5]:


scaler = StandardScaler()
scaled_data = scaler.fit_transform(data_ret)
scaled_data


# In[6]:


scaled_data = pd.DataFrame(scaled_data)
scaled_data


# In[7]:


pca = PCA(n_components=2)
pca.fit(scaled_data)


# In[8]:


pca_data = pca.transform(scaled_data)
print("Explained Variance: ", pca.explained_variance_ratio_)
plt.figure(figsize=(8,6))
plt.scatter(pca_data[:,0], pca_data[:,1])
plt.xlabel('First Principal Component')
plt.ylabel('Second Principal Component')
plt.title('PCA of Stocks')
plt.show()


# In[9]:


correlation_matrix = scaled_data.corr()
correlation_matrix


# In[10]:


eigenvalues = np.linalg.eigvals(correlation_matrix)
eigenvalues


# In[11]:


sorted_eigenvalues = np.sort(eigenvalues)[::-1]  # Reverse the order

# Plotting the eigenvalues
plt.figure(figsize=(8, 6))
plt.bar(range(len(sorted_eigenvalues)), sorted_eigenvalues, color='skyblue')
plt.xlabel('Eigenvalue Index')
plt.ylabel('Eigenvalue Magnitude')
plt.title('Eigenvalues Plot')
plt.show()


# In[12]:


total_variance = np.sum(eigenvalues)
explained_variance_ratio = eigenvalues / total_variance

# Plotting the eigenvalues
plt.figure(figsize=(8, 6))
plt.bar(range(len(explained_variance_ratio)), explained_variance_ratio, color='skyblue')
plt.ylabel('Percentage')
plt.title('Cursory Analysis')
plt.show()


# In[13]:


cov_matrix = np.cov(scaled_data, rowvar=False)

# Compute eigenvalues and eigenvectors
eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

# Select top eigenportfolio (e.g., first eigenvector)
top_eigenvector = eigenvectors[:, 0]  # Replace '0' with the index of the desired eigenvector

# Construct eigenportfolio by normalizing weights
eigenportfolio = top_eigenvector / np.sum(top_eigenvector)
eigenportfolio


# In[14]:


eigenportfolio_returns = np.dot(data_ret, eigenportfolio)
eigenportfolio_returns


# In[15]:


cumulative_return = np.cumprod(1 + eigenportfolio_returns) - 1

cumulative_return = pd.DataFrame(cumulative_return)
cumulative_return


# In[16]:


plt.figure(figsize=(8, 6))
plt.plot(cumulative_return)
plt.xlabel('Date')
plt.ylabel('Cumulated Return')
plt.title('Eigenportfolio Cumulated Return')
plt.show()


# In[ ]:




