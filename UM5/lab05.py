#!/usr/bin/env python
# coding: utf-8

# # Exercise 1
# 

# In[1]:


from sklearn import datasets
import numpy as np
import pandas as pd
data_breast_cancer = datasets.load_breast_cancer(as_frame=True)
#print(data_breast_cancer['DESCR'])


# In[2]:


X=data_breast_cancer["data"][["mean texture","mean symmetry"]]
y=data_breast_cancer["target"]


# In[3]:


from sklearn.model_selection import train_test_split
# X=X.reshape(-1, 1)
# y=y.reshape(-1, 1)


X=X.to_numpy().reshape(-1, 2)
y=y.to_numpy().reshape(-1, 1)

X_train, X_test, y_train, y_test=train_test_split(X, y, test_size=0.2)


# In[4]:


from sklearn.metrics import f1_score
from sklearn.tree import DecisionTreeClassifier


# In[5]:


rnd_clf_max = DecisionTreeClassifier(max_depth=1)
rnd_clf_max.fit(X_train, y_train)
f1_test=f1_score(rnd_clf_max.predict(X_test), y_test)
maxd_test=1
for i in range(2, 1000):
    rnd_clf = DecisionTreeClassifier(max_depth=i)
    rnd_clf.fit(X_train, y_train)
    if f1_test<f1_score(rnd_clf.predict(X_test), y_test):
        f1_test=f1_score(rnd_clf.predict(X_test), y_test)
        maxd_test=i
        rnd_clf_max=rnd_clf


# In[6]:


f1_train=f1_score(rnd_clf_max.predict(X_train), y_train)


# In[7]:


print(f1_test, maxd_test, f1_train, f1_score(rnd_clf_max.predict(X_test), y_test))


# In[8]:


from sklearn.tree import export_graphviz
import graphviz



eg=export_graphviz(rnd_clf_max, out_file=None)
png = graphviz.Source(eg)
png.render("bc.png")


# In[ ]:


import pickle as pkl
from sklearn.metrics import accuracy_score

with open("f1acc_tree.pkl", "wb") as f:
    pkl.dump(
    [maxd_test, f1_train, f1_test,
    accuracy_score(y_train, rnd_clf_max.predict(X_train)),
    accuracy_score(y_test,  rnd_clf_max.predict(X_test))
    ]
    ,f)


# # Exercise 2

# In[ ]:


import numpy as np
import pandas as pd


# In[ ]:


size = 300
X = np.random.rand(size)*5-2.5
w4, w3, w2, w1, w0 = 1, 2, 1, -4, 2
y = w4*(X**4) + w3*(X**3) + w2*(X**2) + w1*X + w0 + np.random.randn(size)*8-4
df = pd.DataFrame({'x': X, 'y': y})
df=df.sort_values(by='x', ignore_index=True)

df.plot.scatter(x='x',y='y')


# In[ ]:


X_train, X_test, y_train, y_test=train_test_split(df['x'].to_numpy().reshape(-1,1), df['y'].to_numpy().reshape(-1,1), test_size=0.2)


# In[ ]:


from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeRegressor

rnd_clf_max_2=DecisionTreeRegressor(max_depth=1)
rnd_clf_max_2.fit(X_train, y_train)
MSE_test=mean_squared_error(y_test, rnd_clf_max_2.predict(X_test))
max_depth_2=1
for i in range (2,1000):
    rnd_clf=DecisionTreeRegressor(max_depth=i)
    rnd_clf.fit(X_train, y_train)
    if MSE_test>mean_squared_error(y_test, rnd_clf.predict(X_test)):

        MSE_test=mean_squared_error(y_test, rnd_clf.predict(X_test))
        max_depth_2=i
        rnd_clf_max_2=rnd_clf


# In[ ]:


#print (max_depth_2)


# In[ ]:


eg=export_graphviz(rnd_clf_max_2, out_file=None)
png = graphviz.Source(eg)
png.render("reg.png")


# In[ ]:


with open("mse_tree.pkl",  "wb") as f:
    pkl.dump([
        max_depth_2,
        mean_squared_error(y_train, rnd_clf_max_2.predict(X_train)),
        MSE_test
    ],f)

