#!/usr/bin/env python
# coding: utf-8

# In[1]:


temp_folder='/home/kate/Research/Property/Notebooks/Experiments/tmp/'
#Experiment_name must NOT contain underscore (_)
Experiment_name='BasicPoisson'
#Experiments log file
Experiments_file='/home/kate/Research/Property/Notebooks/Experiments/Logs/Set1-Poisson.xlsx'
#AllExperiments_tab is a table with a list of all experiments included in the log
#Mandatory columns: Experiment (Experiment_name), Dataset(data file name), Target(target column name from Dataset)
#The rest of the columns are not use in the code below. I usually add in a free form: objective,status,result,notebook name used to conduct the experiment
AllExperiments_tab='Experiments'
#Experiment configuration:
#1.Experiment_Features_tab: differenet datasets to try
#each line in the tab contains a model name and set of features to built a dataset for SageMaker
#a feature can be an exact column name from the Dataset column in AllExperiments_tab or a calculation based on exact column names and eval pandas function
#if the experiment objective is to try different parameters sets, all models (if more then 1) can have the same feature sets.
Experiment_Features_tab='%s Features'%Experiment_name
#2. Alternatively a set of data files with preprocessed data in S3 can be provided in a form:
#Model,Training_data,Validation_data[, Testing_data, Testing_labels]
Experiment_InputData_tab='%s InputData'%Experiment_name
#3. Experiment_Params_tab: each line in the tab contains a model name and set of XGBoost parametersto apply to a model
#the set of models should be consistent in Experiment_Features_tab and Experiment_Params_tab
#parameters can be the same for all models or specific in each model
Experiment_Params_tab='%s Params'%Experiment_name




path_to_data='/home/kate/Research/Property/Data/'
path_to_models='/home/kate/Research/Property/Models/Experiments/%s/'%Experiment_name

path_to_training_data='/home/kate/Research/Property/Data/Experiments/%s/training/'%Experiment_name
path_to_testing_data='/home/kate/Research/Property/Data/Experiments/%s/testing/'%Experiment_name



#preprocessing parameters - the year to separate test data
split_year=2019



#number of folds for CV
num_folds=10



#level of details returning from CV
#any Y return models from a best iteration
#FeatureImportance Y/N
GetFIFlg='Y'
#Scores for Test data (should be provided in fit "test" input) Y/N
GetTestScoreFlg='Y'
#Prediction of Test data (should be provided in fit "test" input) Y/N
GetTestPredFlg='Y'  

score='poisson-nloglik' #'gini'


#Significance level for t-test
alpha=0.05

#n2/n1 (validation/training) ratio for corrected t-test if n2=n1 or n2/n1 = 1 then it's just usual Student t-test withoot correction
#10 folds means 1/9 validation/training ratio 
n2=1
n1=9


# In[2]:


import sys
import time
import os

import re

import pandas as pd
import numpy as np

import xgboost as xgb
import pickle as pkl

#for analyzing results: charts and t-test
import scipy.stats as stats
from matplotlib import pyplot as plt

import warnings
warnings.filterwarnings('ignore')


# ## Experiment
# Experiment is configured in an experiment log file (Excel file, in my case,  in different tabs)

# 1. Reading an experiment configuration (Experiment_name) from an experiment log file (Experiments_file). Target and Dataset columns in AllExperiments_tab contain data file name used and target column

# In[3]:


experiments = pd.read_excel(open(Experiments_file, 'rb'), sheet_name=AllExperiments_tab)


# In[4]:


target=experiments[experiments['Experiment']==Experiment_name]['Target'].values[0]
print('Target of models in %s experiment is %s'%(Experiment_name,target))
data_file=experiments[experiments['Experiment']==Experiment_name]['Dataset'].values[0]
print('Datafile used in %s experiment is %s'%(Experiment_name,data_file))


# 2. Models based on individual datasets to be created, trained and compared in the experiment (Experiment_Features_tab) is a table with first column Model name (should be unique) and next columns [1:51] features to train the model. Feature is the exact column name from the dataset or a calculation based on exact column names and eval pandas function
# 
# This configuration will be used to preprocess data and also need to be moved to S3 in csv format for easy reading in a preprocessing script if we use AWS SKLearnProcessor/job/instances

# In[5]:


model_features = pd.read_excel(open(Experiments_file, 'rb'), sheet_name=Experiment_Features_tab)
model_features  


# 2a.Preprocessed data may already exists in an S3. Experiment configuration can provide the list of files per model. In this case (len(preprocessed_data)==0) the code skips all steps to preprocess data

# In[6]:


try:
    preprocessed_data = pd.read_excel(open(Experiments_file, 'rb'), sheet_name=Experiment_InputData_tab)
    #preprocessed_data = pd.concat([preprocessed_data,model_features.drop('Model',axis=1)], axis=1)
except:
    preprocessed_data = pd.DataFrame()


# 3. Model params to be used in training is a table with first column Model name (should be unique and corresponds to models in Experiment_Features_tab) and next columns are XGBoost parameters
# In a general case, all models can have the same parameters

# In[7]:


model_params = pd.read_excel(open(Experiments_file, 'rb'), sheet_name=Experiment_Params_tab)
model_params


# 4.Verification if we have the same set of models in both configurations

# In[8]:


models_from_model_features=model_features['Model'].tolist()
models_from_model_params=model_params['Model'].tolist()
if len([x for x in models_from_model_features if x not in models_from_model_params])!=0:
    raise Exception('Different set of models in featuresets and parametersets!')
if len(preprocessed_data)>0:
    models_from_preprocessed_data=preprocessed_data['Model'].tolist()
    if len([x for x in models_from_preprocessed_data if x not in models_from_model_params])!=0:
        raise Exception('Different set of models in input data and parametersets!')


# In[9]:


#sys.path.append('/home/kate/Research/YearBuilt/Notebooks/Experiments')
import ExperimentsUtils as eu


# ## Data preprocessing

# Preprocessing output (training and testing datasets) are saved separately for each model in a folder with the same name as a models name configured in the experiment

# In[10]:


if len(preprocessed_data)==0:
    preprocessed_data = pd.DataFrame(columns=['Model', 'Training_data',  'Testing_data', 'Training_offset','Testing_offset'])
    
    input_data_path=path_to_data+data_file
    print('Reading input data from {}'.format(input_data_path))
    dataset = pd.read_csv(input_data_path, error_bad_lines=False, index_col=False)
    dataset_test=dataset[(dataset.cal_year == split_year)]
    dataset=dataset[(dataset.cal_year < split_year)]    
    
 
    

    #iterating thru config file with models and featureset
    feature_columns=model_features.columns.tolist()
    feature_columns.remove('Offset')
    feature_columns.remove('Model')
    feature_columns
    for index, row in model_features.iterrows():
        model=row['Model']
        print (index, ': Creating datasets for model %s'%model)
        featureset=row[feature_columns].tolist()

        
        featureset=[x for x in featureset if str(x) != 'nan']
        print(','.join(featureset))
        
        #creating dataset for a model according to configured dataset
        X = pd.DataFrame()
        X_test = pd.DataFrame()        
        for f in featureset:
            X[f]=dataset.eval(f)
            X_test[f]=dataset_test.eval(f)            
        y=dataset.eval(target)
        y_test=dataset_test.eval(target) 
        
        #Offset is not a mandatory column
        offset_flg=False
        test_offset_filename=''
        train_offset_filename=''
        try:
            offset_column=row['Offset']
            if offset_column != 'nan':
                offset_train=dataset.eval(offset_column)
                offset_test=dataset_test.eval(offset_column)            
                offset_flg=True
        except:
            offset_flg=False
        
        
        print('Testing data...')
        
        test_dataset=pd.DataFrame({target:y_test}).join(X_test)
        
        test_data_output_path = path_to_testing_data+model              
        if not os.path.exists(test_data_output_path):
            os.makedirs(test_data_output_path) 
        test_data_filename = os.path.join(test_data_output_path,  'testing_%s.csv'%(model)) 
        test_dataset.to_csv(test_data_filename, header=True, index=False)
        if offset_flg:
            test_offset_filename = os.path.join(test_data_output_path,  'offset_%s.csv'%(model))
            offset_test.to_csv(test_offset_filename, header=True, index=False)
        #The rest of the data will be used in cv-fold as a whole and seprated to training/validation insode cv  
        print('Training data...')
        training_dataset=pd.DataFrame({target:y}).join(X)
 
        train_data_output_path=path_to_training_data+model
        if not os.path.exists(train_data_output_path):
            os.makedirs(train_data_output_path)
        train_data_filename = os.path.join(train_data_output_path, 'training_%s.csv'%model) 
        training_dataset.to_csv(train_data_filename, header=True, index=False)   

        if offset_flg:
            train_offset_filename = os.path.join(train_data_output_path,  'offset_%s.csv'%(model))
            offset_train.to_csv(train_offset_filename, header=True, index=False)
            
        preprocessed_data.loc[index]=[model, train_data_filename,test_data_filename,train_offset_filename,test_offset_filename]
        
    #Saving into the Experiment log file names of created training and validation datasets
    preprocessed_data = pd.concat([preprocessed_data,model_features.drop('Model',axis=1)], axis=1) 
    eu.SaveToExperimentLog(Experiments_file, '%s InputData'%Experiment_name, preprocessed_data)


# In[11]:


preprocessed_data


# ## Model training

# In[12]:


models_from_preprocessed_data=preprocessed_data['Model'].tolist()
models_from_model_params=model_params['Model'].tolist()
if len([x for x in models_from_preprocessed_data if x not in models_from_model_params])!=0:
    raise Exception('Different set of models in preprocessed_data and parametersets!')
#using merge because, in general, we can have different number of rows in each dataframe - folds in data and different sets of params
data_for_training=pd.merge(model_params, preprocessed_data, on='Model', how='inner')
data_for_training


# In[13]:


def cv_misc_callback(oof_train_scores:list, oof_valid_scores:list, best_models:list,  maximize=True):
    """
    It's called inside XGB CV to catch individual folds scores and models
    """    
    state = {}
    def init(env):
        if maximize:
            state['best_score'] = -np.inf
        else:
            state['best_score'] = np.inf 
#--------------------------------------------------------------------------------            
    def callback(env):
        #init env if empty
        if not state:
            init(env)
        best_score = state['best_score']
        score = env.evaluation_result_list[-1][1]
        #extract best model if a current score is better then previous
        if (maximize and score > best_score) or (not maximize and score < best_score):
            for i, cvpack in enumerate(env.cvfolds): 
                best_models[i]=cvpack.bst
            state['best_score'] = score    
        #all iterations individual folds scores
        folds_train_scores = []
        folds_valid_scores = []
        for i, cvpack in enumerate(env.cvfolds):    
            scores = cvpack.eval(iteration=0,feval=feval)
            #print(scores)
            scores_l = re.split(': |\t',scores)
            train_score=scores_l[1].rpartition(':')[2]
            valid_score=scores_l[2].rpartition(':')[2]
            folds_train_scores.append(train_score)
            folds_valid_scores.append(valid_score)
        oof_train_scores.append(folds_train_scores)
        oof_valid_scores.append(folds_valid_scores)
#--------------------------------------------------------------------------------        
    callback.before_iteration = False
    return callback


# In[14]:


def gini(y, pred):
    g = np.asarray(np.c_[y, pred, np.arange(len(y)) ], dtype=np.float)
    g = g[np.lexsort((g[:,2], -1*g[:,1]))]
    gs = g[:,0].cumsum().sum() / g[:,0].sum()
    gs -= (len(y) + 1) / 2.
    return gs / len(y)
def gini_xgb(pred, y):
    y = y.get_label()
    return 'gini', gini(y, pred) / gini(y, y)


# In[15]:


from sklearn.metrics import roc_auc_score


# In[16]:


def nLogLik_XGBoost (act,pred):
    df = pd.DataFrame({"act":act,"pred":pred}) 
    return np.mean( df.pred - df.act*np.log(df.pred))


# In[17]:


#parameters depending on score
#custom evaluation function
feval=gini_xgb if score=='gini' else None
#best model from xgboost CV: with minimum or maximum score
Maximize=False if score=='poisson-nloglik'  else True


# In[18]:


#regexpression to exclude features (F1..F25) from the list of parameters
regex = re.compile('F[ 0-9]')
for index, row in data_for_training.iterrows():
    model='%s-%s'%(row['Model'],index)
    print(model)
    
    #Dataset
    train_dataset = pd.read_csv(row['Training_data'], error_bad_lines=False, index_col=False)
    X_train = train_dataset.iloc[:,1:]
    y_train = train_dataset.iloc[:, 0]
    dtrain = xgb.DMatrix(X_train, y_train)
    #offset is not mandatory
    try:
        train_offset = pd.read_csv(row['Training_offset'], error_bad_lines=False, index_col=False)
        dtrain.set_base_margin(train_offset.values)
        print('training Offset was added')
    except:
        pass
    
    #Test Dataset
    if 'Y' in (GetTestScoreFlg,GetTestPredFlg):
        test_dataset = pd.read_csv(row['Testing_data'], error_bad_lines=False, index_col=False)
        X_test = test_dataset.iloc[:,1:]
        y_test = test_dataset.iloc[:, 0]
        dtest = xgb.DMatrix(X_test, y_test)
        #offset is not mandatory
        try:
            testing_offset = pd.read_csv(row['Testing_offset'], error_bad_lines=False, index_col=False)
            dtest.set_base_margin(testing_offset.values)
            print('testing Offset was added')            
        except:
            pass
    
    #Hyperparameters
    hyperparameters = {     
        'seed': 42       
    } 
    for i, param in enumerate(data_for_training.columns):
        #skip first column with Model name and dataset names or features
        #if do not exclude then they will be added into experiment analytics as parameters but not used in training anyway
        if ((param in ('Model','Training_data','Validation_data','Testing_data','Testing_labels','Offset','Training_offset', 'Testing_offset')) | (bool(re.match(regex, param)))):
            continue
        if param=='num_round':
            continue
        if ((param=='eval_metric') & (score=='gini')):
            hyperparameters['disable_default_eval_metric'] = '1'
            continue       
        hyperparameters[param] = row[param]
    print(hyperparameters)
    num_boost_round = row['num_round']
    early_stopping_rounds = 100
                        
                          
    #OUT parameters from custom callback function: 
    #train and valid scores from all folds
    oof_train_scores = []
    oof_valid_scores = []
    #Best Model
    best_models=[None]*num_folds
    #===========================================================================================================    
    args = {'params':hyperparameters, 
                  'dtrain':dtrain,             
                  'feval':feval,
                  'num_boost_round':num_boost_round,
                  'nfold':num_folds, 
                  'stratified':True, 
                  'shuffle':True,
                  'early_stopping_rounds':early_stopping_rounds, 
                  'seed':42,
                  'callbacks':[cv_misc_callback(oof_train_scores, oof_valid_scores,best_models,Maximize), xgb.callback.print_evaluation(period=10)]}     
    
    cv_results=xgb.cv(**args)

    #===========================================================================================================
    #scores to dataframe
    df_oof_train_scores = pd.DataFrame.from_records(oof_train_scores).apply(pd.to_numeric)
    df_oof_valid_scores = pd.DataFrame.from_records(oof_valid_scores).apply(pd.to_numeric)


    
    #only folds scores columns names
    columns = df_oof_train_scores.columns.tolist()

    
    #mean and std, sem 
    df_oof_train_scores['std'] = df_oof_train_scores[columns].std(axis=1)
    df_oof_valid_scores['std'] = df_oof_valid_scores[columns].std(axis=1)
    df_oof_train_scores['sem'] = df_oof_train_scores[columns].sem(axis=1)
    df_oof_valid_scores['sem'] = df_oof_valid_scores[columns].sem(axis=1)    
    df_oof_train_scores['mean'] = df_oof_train_scores[columns].mean(axis=1)
    df_oof_valid_scores['mean'] = df_oof_valid_scores[columns].mean(axis=1)
    
    #best models feature importance 
    if GetFIFlg=='Y':
        oof_fi_weight_best = {}
        oof_fi_gain_best = {}
        oof_fi_cover_best = {}
        for i in range(0,num_folds):
            oof_fi_weight_best[i]=best_models[i].get_score(importance_type='weight')
            oof_fi_gain_best[i]= best_models[i].get_score(importance_type='gain')
            oof_fi_cover_best[i]= best_models[i].get_score(importance_type='cover')
    
        #converting to dataframe
        df_oof_fi_weight_best = pd.DataFrame(oof_fi_weight_best).apply(pd.to_numeric)
        df_oof_fi_gain_best = pd.DataFrame(oof_fi_gain_best).apply(pd.to_numeric)
        df_oof_fi_cover_best = pd.DataFrame(oof_fi_cover_best).apply(pd.to_numeric)
    
    
        #mean and std, sem 
        df_oof_fi_weight_best['std'] = df_oof_fi_weight_best[columns].std(axis=1)
        df_oof_fi_gain_best['std'] = df_oof_fi_gain_best[columns].std(axis=1)
        df_oof_fi_cover_best['std'] = df_oof_fi_cover_best[columns].std(axis=1)
    
        df_oof_fi_weight_best['sem'] = df_oof_fi_weight_best[columns].sem(axis=1)
        df_oof_fi_gain_best['sem'] = df_oof_fi_gain_best[columns].sem(axis=1)
        df_oof_fi_cover_best['sem'] = df_oof_fi_cover_best[columns].sem(axis=1)
    
        df_oof_fi_weight_best['mean'] = df_oof_fi_weight_best[columns].mean(axis=1)
        df_oof_fi_gain_best['mean'] = df_oof_fi_gain_best[columns].mean(axis=1)
        df_oof_fi_cover_best['mean'] = df_oof_fi_cover_best[columns].mean(axis=1)
    
    
        #feature codes from index to column
        df_oof_fi_weight_best.reset_index(level=0, inplace=True)
        df_oof_fi_weight_best.columns=['feature'] + columns + ['std','sem','mean']
        df_oof_fi_gain_best.reset_index(level=0, inplace=True)
        df_oof_fi_gain_best.columns=['feature'] + columns + ['std','sem','mean']
        df_oof_fi_cover_best.reset_index(level=0, inplace=True)
        df_oof_fi_cover_best.columns=['feature'] + columns + ['std','sem','mean']
    
    if 'Y' in (GetTestScoreFlg,GetTestPredFlg):
        #Prediction on test data  from folds best models...
        df_prediction=pd.DataFrame()
        df_prediction['actual']=dtest.get_label()
        for i in range(0,num_folds):
            df_prediction[i]=best_models[i].predict(dtest)
   
        #Test scores from test prediction   
        df_scores = pd.DataFrame()
        for i in range(0,num_folds):
            if score=='gini':
                df_scores[i]=[gini(df_prediction['actual'], df_prediction[i])/gini(df_prediction['actual'], df_prediction['actual'])]
            elif score=='AUC':
                df_scores[i]=[roc_auc_score(df_prediction['actual'], df_prediction[i])]
            elif score=='poisson-nloglik':
                df_scores[i]=[nLogLik_XGBoost(df_prediction['actual'],df_prediction[i])]
                

        df_scores['std'] = df_scores[columns].std(axis=1)
        df_scores['sem'] = df_scores[columns].sem(axis=1)
        df_scores['mean'] = df_scores[columns].mean(axis=1)

    output_data_dir=path_to_models+model
    if not os.path.exists(output_data_dir):
        os.makedirs(output_data_dir)
            
        
    for i in range(0,num_folds):
        model_location = os.path.join(output_data_dir , 'model-fold-'+str(i))
        pkl.dump(best_models[i], open(model_location, 'wb'))
        
    if  GetTestPredFlg=='Y':    
        predictions_location = os.path.join(output_data_dir, 'test_predictions.csv')
        print('Saving test predictions at {}'.format(predictions_location))            
        df_prediction.to_csv(predictions_location, header=True, index=False)
        
    if  GetTestScoreFlg=='Y':
        oof_test_scores_location = os.path.join(output_data_dir, 'oof_test_scores.csv')
        print('Saving oof_test_scores at {}'.format(oof_test_scores_location))
        df_scores.to_csv(oof_test_scores_location, header=True, index=False)
        
    cv_result_location = os.path.join(output_data_dir, 'cv_results.csv')
    print('Saving cv results at {}'.format(cv_result_location))
    cv_results.to_csv(cv_result_location, header=True, index=False)
        
    oof_train_scores_location = os.path.join(output_data_dir, 'oof_train_scores.csv')
    print('Saving oof_train_scores at {}'.format(oof_train_scores_location))
    df_oof_train_scores.to_csv(oof_train_scores_location, header=True, index=False)  
        
    oof_valid_scores_location = os.path.join(output_data_dir, 'oof_valid_scores.csv')
    print('Saving oof_valid_scores at {}'.format(oof_valid_scores_location))
    df_oof_valid_scores.to_csv(oof_valid_scores_location, header=True, index=False)
        
    if  GetFIFlg=='Y':
        oof_fi_weight_best_location = os.path.join(output_data_dir, 'oof_fi_weight_best.csv')
        print('Saving oof_fi_weight_best at {}'.format(oof_fi_weight_best_location))
        df_oof_fi_weight_best.to_csv(oof_fi_weight_best_location, header=True, index=False)  
        
        oof_fi_gain_best_location = os.path.join(output_data_dir, 'oof_fi_gain_best.csv')
        print('Saving oof_fi_gain_best at {}'.format(oof_fi_gain_best_location))
        df_oof_fi_gain_best.to_csv(oof_fi_gain_best_location, header=True, index=False)        
        
        oof_fi_cover_best_location = os.path.join(output_data_dir, 'oof_fi_cover_best.csv')
        print('Saving oof_fi_cover_best at {}'.format(oof_fi_cover_best_location))
        df_oof_fi_cover_best.to_csv(oof_fi_cover_best_location, header=True, index=False)     