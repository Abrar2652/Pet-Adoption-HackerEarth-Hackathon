#!/usr/bin/env python
# coding: utf-8

# In[142]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# In[184]:


df = pd.read_csv('./ML hackathon/Dataset/train.csv',encoding = "utf-8")
df = df.fillna(3)
df


# In[192]:


df1 = pd.read_csv('./ML hackathon/Dataset/train.csv',encoding = "utf-8")
df1 = df1.fillna(3)
df2 = pd.read_csv('./ML hackathon/Dataset/test.csv',encoding = "utf-8")
df2 = df2.fillna(3)
df3 = pd.concat([df1,df2])
df3


# In[193]:


df2


# In[199]:


def get_ready(df):
    from sklearn.model_selection import train_test_split
    # declare features to use
    features = ['length(m)','height(cm)','color_type','X1','X2','condition']
    # create copy
    df_copy = df3.copy()
    df_copy['target'] = df_copy['pet_category'] 
    # separate features from target
    X=df_copy[features]
    y=df_copy['target']
    # perform split and return
    return train_test_split(X, y, test_size=0.3,shuffle=False )


# In[200]:


X_train, X_test, y_train, y_test = get_ready(df3)


# In[201]:


from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.linear_model import SGDClassifier

    
def train_classifier(X_train, y_train):
   
    classifier1 = RandomForestClassifier()
    classifier1.fit(X_train, y_train)
    return classifier1


# In[202]:


v = train_classifier(X_train, y_train)
y_pred = v.predict(X_test)
y_pred


# In[203]:


f['pet_id'] = df2['pet_id']
f = pd.DataFrame(data=y_pred, columns=["pet_category"])

f.to_csv('sub1.csv')


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




