
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
import math

# In[11]:


#Normalized Weighted Gini
#https://www.kaggle.com/c/liberty-mutual-fire-peril/discussion/9880
#http://blog.nguyenvq.com/blog/2015/09/25/calculate-the-weighted-gini-coefficient-or-auc-in-r/
def WeightedGini(act,pred,weight): 
    df = pd.DataFrame({"act":act,"pred":pred,"weight":weight}) 
    df = df.sort_values('pred',ascending=False) 
    df["random"] = (df.weight / df.weight.sum()).cumsum() 
    total_pos = (df.act * df.weight).sum()
    df["cumposfound"] = (df.act * df.weight).cumsum()
    df["lorentz"] = df.cumposfound / total_pos
    n = df.shape[0]
    gini = sum(df.lorentz[1:].values * (df.random[:-1])) - sum(df.lorentz[:-1].values * (df.random[1:]))
    return gini

def NormalizedWeightedGini(act,pred,weight):
    return WeightedGini(act,pred,weight) / WeightedGini(act,act,weight)
#Test
#var11 = [1, 2, 5, 4, 3] 
#pred = [0.1, 0.4, 0.3, 1.2, 0.0]
#act = [0, 0, 1, 0, 1]
#normalizedweightedgini(act,pred,var11)
#-0.821428571428572
#---------------------------------------------------
def nLogLik_XGBoost (act,pred):
    df = pd.DataFrame({"act":act,"pred":pred}) 
    return np.mean( df.pred - df.act*np.log(df.pred))
#----------------------------------------------------
def mae(y, pred):
    return mean_absolute_error(pred, y)
def rmse(y, pred):
    return math.sqrt(mean_squared_error(pred, y))
