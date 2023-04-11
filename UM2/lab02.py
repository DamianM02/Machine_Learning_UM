#!/usr/bin/env python
# coding: utf-8

# In[33]:


import pandas as pd

from sklearn.datasets import fetch_openml
mnist=fetch_openml('mnist_784', version=1)

#import numpy as np


# In[34]:


import numpy as np
#print((np.array(mnist.data.loc[56005]).reshape(28,28)>0).astype(int))
pass


# In[35]:


X, y = mnist["data"], mnist["target"].astype(np.uint8)


# In[35]:





# In[36]:


mnist.keys()


# In[36]:





# In[37]:


X_train, X_test=X[:56000], X[56000:]
y_train, y_test=y[:56000], y[56000:]


# In[38]:


y_train_0=(y_train==0)
y_test_0=(y_test==0)


# In[39]:


print(y_train_0)
print(np.unique(y_train_0))
print(len(y_train_0))


# In[40]:


from           sklearn.linear_model           import           SGDClassifier

sgd_clf=SGDClassifier(random_state=42)
sgd_clf.fit(X_train, y_train_0)


# In[41]:


xp=sgd_clf.predict(X_train)
a=0
b=0

for i in range(0, 56000):
    b+=1
    if y_train_0[i]==xp[i]:
        a+=1

print (a/b, "  ", b)

list=[]
list.append(a/b)




xp=sgd_clf.predict(X_test)
a=0
b=0

for i in range(56000, 70000):
    b+=1
    if y_test_0[i]==xp[i-56000]:
        a+=1

print (a/b)
print (b)

list.append(a/b)









# In[42]:


# a=0;
# xp=sgd_clf.predict(X_test)[i-56000 :]
# for i in range(56000, 70000):
#     if y_test_0[i]!=xp[i-56000]:
#         a+=1

# print(a)


# In[43]:


from numpy import ndarray
from           sklearn.model_selection           import           cross_val_score
import time
start=time.time()
score = ndarray((3, ), buffer=cross_val_score(sgd_clf, X_train, y_train_0, cv=3, scoring="accuracy", n_jobs=-1))

print(time.time()-start)
print(score)


# In[44]:


import pickle as pi

with open('sgd_cva.pkl', 'wb') as f:
    pi.dump(score, f)


# In[45]:


with open('sgd_acc.pkl', 'wb') as f:
    pi.dump(list, f)


# In[46]:


# from sklearn.svm import SVC
# svm_clf=SVC()
# svm_clf.fit(X_train, y_train)


# In[47]:


#xpp=svm_clf.predict(X_test)

# for i in range(56000, 70000):
#     print(y_test[i],  "  ", xpp[i-56000])


# In[48]:


sgd_m_clf=SGDClassifier(random_state=42, n_jobs=-1)
sgd_m_clf.fit(X_train, y_train)


# In[49]:


# xpp=sgd_m_clf.predict(X_test)
#
# for i in range(56000, 70000):
#     print(y_test[i],  "  ", xpp[i-56000])


# In[50]:


y_test_pred=cross_val_score(sgd_m_clf, X_test, y_test, cv=3, n_jobs=-1)


# In[51]:


from sklearn.metrics import confusion_matrix

y_test_predict=sgd_m_clf.predict(X_test)
conf_mx=confusion_matrix(y_test, y_test_predict)
print(conf_mx)


# In[52]:


with open("sgd_cmx.pkl", "wb") as f:
    pi.dump(conf_mx, f)


# In[52]:




