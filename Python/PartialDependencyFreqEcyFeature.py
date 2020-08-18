
# coding: utf-8

# In[1]:


ModelsDir = '/home/kate/Research/Property/Models/'
DataDir = '/home/kate/Research/Property/Data/'


# In[3]:


# from https://xiaoxiaowang87.github.io/monotonicity_constraint/
def partial_dependency(model, X,  feature):

    """
    Calculate the dependency (or partial dependency) of a response variable on a predictor (or multiple predictors)
    1. Sample a grid of values of a predictor for numeric continuous or all unique values for categorical or discrete continuous.
    2. For each value, replace every row of that predictor with this value, calculate the average prediction.
    """

    X_temp = X.copy()
    

    if feature in ['yearbuilt','sqft','water_risk_3_blk','ecy']:
        # continuous
        grid = np.linspace(np.percentile(X_temp[feature], 0.1),
                       np.percentile(X_temp[feature], 99.5),
                           50)
    else:
        #categorical
        grid = X_temp[feature].unique()

    y_pred = np.zeros(len(grid))

    for i, val in enumerate(grid):
        X_temp[feature] = val
        d_temp=xgb.DMatrix(X_temp.values)
        #d_temp.set_base_margin(offset.values)
        y_pred[i] = np.average(model.predict(d_temp,ntree_limit=model.best_ntree_limit+50))


    return grid, y_pred


# In[4]:


import pandas as pd
import numpy as np
import pickle
import xgboost as xgb
import os


# In[ ]:


kfold = 5


# In[5]:


training_dataset = pd.read_csv('%sproperty_wcf_class_training.csv'%DataDir, error_bad_lines=False, index_col=False)


# In[7]:


featureset  = [
'roofcd_encd',
'sqft',  
'usagetype_encd',
'yearbuilt',
'water_risk_3_blk',
'landlordind',
'multipolicyind',    
'cova_deductible',
'cova_limit',
'ecy'
        ]


# In[8]:


pd_featureset  = [
'roofcd_encd',
'sqft',  
'usagetype_encd',
'yearbuilt',
'water_risk_3_blk',
'landlordind',
'multipolicyind',    
'cova_deductible',
'cova_limit',
'ecy'
        ]



# In[9]:


X = training_dataset.loc[:,featureset]


# In[11]:


Model='wc_class_f_ecy_XGB'
all_fm_pd = pd.DataFrame()
for i in range(0,kfold):
    ModelName='%s_%s'%(Model,i)
    print('Processing Model: %s'%ModelName)
    xgb_model_file='%s%s.model'%(ModelsDir,ModelName)
    xgb_model = pickle.load(open(xgb_model_file, 'rb'))
    for f in pd_featureset:
        print('Processing:%s'%f)
        grid, y_pred = partial_dependency(xgb_model,X,f)
        fm_pd=pd.concat([pd.Series(grid), pd.Series(y_pred)], axis=1)
        fm_pd.columns=['value','pd']
        fm_pd['feature']=f
        fm_pd['fold']=i
        fm_pd['ModelName']=Model
        all_fm_pd=all_fm_pd.append(fm_pd)
        all_fm_pd.to_csv('%s%s_PartialDependency.csv'%(ModelsDir,Model),header=True,index=False);


# In[ ]:


training_dataset = pd.read_csv('%sproperty_wcf_training.csv'%DataDir, error_bad_lines=False, index_col=False)


# In[ ]:


X = training_dataset.loc[:,featureset]


# In[ ]:


Model='wc_Poisson_f_ecy_XGB'
all_fm_pd = pd.DataFrame()
for i in range(0,kfold):
    ModelName='%s_%s'%(Model,i)
    print('Processing Model: %s'%ModelName)
    xgb_model_file='%s%s.model'%(ModelsDir,ModelName)
    xgb_model = pickle.load(open(xgb_model_file, 'rb'))
    for f in pd_featureset:
        print('Processing:%s'%f)
        grid, y_pred = partial_dependency(xgb_model,X,f)
        fm_pd=pd.concat([pd.Series(grid), pd.Series(y_pred)], axis=1)
        fm_pd.columns=['value','pd']
        fm_pd['feature']=f
        fm_pd['fold']=i
        fm_pd['ModelName']=Model
        all_fm_pd=all_fm_pd.append(fm_pd)
        all_fm_pd.to_csv('%s%s_PartialDependency.csv'%(ModelsDir,Model),header=True,index=False);

