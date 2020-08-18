
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import pickle
import xgboost as xgb
import os
import shap


# In[2]:


ModelsDir = '/home/kate/Research/Property/Models/'
DataDir = '/home/kate/Research/Property/Data/'


# In[3]:


Models = ['wc_Linear_Reg_XGB_mae',
'wc_LogRegObj_Reg_XGB_mae',
'wc_Gamma_Reg_XGB_mae']


# In[4]:


prediction_dataset = pd.read_csv('%sproperty_water_claims_non_cat_fs.csv'%DataDir, error_bad_lines=False, index_col=False)


# In[5]:


featureset_shap  = [
'cova_deductible_shap_value',
'roofcd_encd_shap_value',
'water_risk_sev_3_blk_shap_value',
'sqft_shap_value',
'rep_cost_3_blk_shap_value',
'yearbuilt_shap_value',
'ecy_shap_value',
'usagetype_encd_shap_value'   
]


# In[6]:


featureset  = [
'cova_deductible',
'roofcd_encd',
'water_risk_sev_3_blk',
'sqft',
'rep_cost_3_blk',
'yearbuilt',
'ecy',
'usagetype_encd'
]


# In[7]:


kfold = 5


# In[9]:


#
X_pred=prediction_dataset[featureset]
Dpred = xgb.DMatrix(X_pred.values)


# ## Creating shap values for testing, prediction and training datasets

# In[10]:


dataset_shap_values = pd.DataFrame()


# In[11]:


for Model in Models:
    for i in range(0,kfold):
        dataset_shap_values = pd.DataFrame()
        ModelName=Model+"_%s"%i
        xgb_model_file='%s%s.model'%(ModelsDir,ModelName)
        print('Processing model %s, fold %s...'%(Model,i))
        xgb_model = pickle.load(open(xgb_model_file, 'rb'))
        explainer = shap.TreeExplainer(xgb_model)
        #Prediction dataset explaining
        shap_values = explainer.shap_values(Dpred)
        df_shap_values = pd.DataFrame(data=shap_values,   columns=featureset_shap)
        df_shap_values['original_output_value'] = df_shap_values.sum(axis=1)
        df_shap_values['expected_value'] = explainer.expected_value
        df_shap_values['output_value'] = df_shap_values['expected_value'] + df_shap_values['original_output_value']
        df_shap_values['modeldata_id'] = prediction_dataset['modeldata_id']
        df_shap_values['cal_year'] = prediction_dataset['cal_year']
        df_shap_values['ModelName'] = Model
        df_shap_values['fold'] = i
        df_shap_values = df_shap_values[['ModelName','fold','modeldata_id','cal_year']+featureset_shap+['original_output_value','expected_value','output_value']]
        dataset_shap_values = dataset_shap_values.append(df_shap_values)          
        #Saving 
        dataset_shap_values.to_csv('%sseverity_shap_values_%s.csv'%(DataDir,ModelName),header=True,index=False)

