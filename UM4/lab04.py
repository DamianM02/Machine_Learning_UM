#!/usr/bin/env python
# coding: utf-8

# In[190]:


from sklearn import datasets
import numpy as np


# In[191]:


df= datasets.load_breast_cancer(as_frame=True)
#print(df['DESCR'])


# In[192]:


import pandas as pd
df["data"].keys()


# In[193]:


X=(df["data"][["mean area","mean smoothness"]])
y=(df["target"]==1).astype(np.uint8)


# In[194]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test=train_test_split(X, y, test_size=0.2, random_state=42)


# In[195]:


from sklearn.svm import LinearSVC
svc_clf_1=LinearSVC(loss="hinge", random_state=42)
svc_clf_1.fit(X_train, y_train)


# In[196]:


from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
svc_clf_2=Pipeline([
    ("scaler", StandardScaler()),
    ("linear_svc", LinearSVC(loss="hinge", random_state=42))
])
svc_clf_2.fit(X_train, y_train)


# In[197]:


from sklearn.metrics import accuracy_score
list=[
    accuracy_score(y_train, svc_clf_1.predict(X_train)),
    accuracy_score(y_test,  svc_clf_1.predict(X_test)),
    accuracy_score(y_train, svc_clf_2.predict(X_train)),
    accuracy_score(y_test,  svc_clf_2.predict(X_test))
]


# In[198]:


print(list)


# In[199]:


import pickle as pk
with open("bc_acc.pkl", 'wb') as f:
    pk.dump(list, f)


# In[200]:


data_iris = datasets.load_iris(as_frame=True)
#print(data_iris['DESCR'])


# In[201]:


data_iris["data"].keys()


# In[202]:


X=data_iris["data"][["petal length (cm)","petal width (cm)"]]
y=data_iris["target"]


# In[203]:


(X_train, X_test, y_train, y_test )= train_test_split(X, y,test_size=0.2, random_state=436)


# In[204]:


#print(len(X_train), len(X_test), len(y_train), len(y_test))


# In[205]:


svc_clf_1=LinearSVC(loss="hinge", random_state=42)
svc_clf_1.fit(X_train, y_train)


# In[206]:


svc_clf_2=Pipeline(
    [
        ("", StandardScaler()),
        (" ",LinearSVC(loss="hinge",random_state=42))
    ]
)
svc_clf_2.fit(X_train, y_train)


# In[207]:


list2=[
    accuracy_score(y_train, svc_clf_1.predict(X_train)),
    accuracy_score(y_test,  svc_clf_1.predict(X_test)),
    accuracy_score(y_train, svc_clf_2.predict(X_train)),
    accuracy_score(y_test,  svc_clf_2.predict(X_test))
]


# In[208]:


print(list2)


# In[209]:


with open("iris_acc.pkl", "wb") as f:
    pk.dump(list2, f)


# In[209]:




