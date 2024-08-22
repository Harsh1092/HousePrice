#!/usr/bin/env python
# coding: utf-8

# In[29]:


import pandas as pd
import numpy as np


# In[30]:


df = pd.read_csv("Delhi house data.csv")


# In[31]:


df.head()


# In[32]:


df = df.drop(columns=['Furnishing', 'Locality', 'Status', 'Transaction', 'Type', 'Per_Sqft'])


# In[33]:


df.head()


# In[34]:


df.isna().sum()


# In[35]:


df['Bathroom']= df['Bathroom'].fillna(df['Bathroom'].median())


# In[36]:


df['Parking'] = df['Parking'].fillna(df['Parking'].median())


# In[37]:


df.describe()


# In[38]:


df.isna().sum()


# In[39]:


df.to_csv("dataset.csv")


# In[46]:


X=df.drop(columns=['Price'])
y=df['Price']


# In[52]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression,Lasso,Ridge
from sklearn.metrics import accuracy_score, r2_score
lr = LinearRegression()


# In[49]:


X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=0)
print(X_train.shape)
print(X_test.shape)


# In[53]:


lr.fit(X_train, y_train)
y_pred = lr.predict(x_test)


# In[55]:


r2_score(y_test, y_pred)


# In[ ]:




