import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn import preprocessing
import random
import math
#import csv
#import time
from datetime import datetime

def SettingsForXGBoost( eta = 0.2, gamma = 1, min_child_weight = 6, 
                        max_depth = 30, max_delta_step = 2):
    params = {}
    params["booster"] = "gbtree"
    #params["booster"] = "gblinear"
    params["objective"] = "reg:linear"
    params["bst:eta"] = eta
    params["bst:gamma"] = gamma
    #params["lambda"] = lambda1    
    params["bst:min_child_weight"] = min_child_weight
    #params["subsample"] = subsample
    #params["colsample_bytree"] = colsample_bytree
    #params["scale_pos_weight"] = scale_pos_weight
    params["silent"] = 1
    params["bst:max_depth"] = max_depth
    params["bst:max_delta_step"] = max_delta_step
    params["nthread"] = 16
    plst = list(params.items())
    return params, plst

if __name__ == "__main__":
    startTime = datetime.now()
    predict = True
    csvmap = {}
    print 'loading'
    csvmap['train'] = pd.read_csv('../Data/train.csv', parse_dates=[2,], low_memory=False)
    csvmap['stores'] = pd.read_csv('../Data/store.csv', low_memory=False)
    if(predict):
        csvmap['test'] = pd.read_csv('../Data/test.csv', low_memory=False, parse_dates=[3,])
    print 'Getting date info'
    csvmap['train']['year'] = pd.DatetimeIndex(csvmap['train']['Date']).year
    csvmap['train']['month'] = pd.DatetimeIndex(csvmap['train']['Date']).month
    csvmap['train']['week'] = pd.DatetimeIndex(csvmap['train']['Date']).week
    csvmap['train']['day'] = pd.DatetimeIndex(csvmap['train']['Date']).day
    
    csvmap['test']['year'] = pd.DatetimeIndex(csvmap['test']['Date']).year
    csvmap['test']['month'] = pd.DatetimeIndex(csvmap['test']['Date']).month
    csvmap['test']['week'] = pd.DatetimeIndex(csvmap['test']['Date']).week
    csvmap['test']['day'] = pd.DatetimeIndex(csvmap['test']['Date']).day
    
    csvmap['train'] = pd.merge(csvmap['train'], csvmap['stores'], on='Store')
    csvmap['test'] = pd.merge(csvmap['test'], csvmap['stores'], on='Store')
    
    csvmap['train'] = csvmap['train'].drop('Date',1)
    csvmap['test'] = csvmap['test'].drop('Date',1)
    
    print 'Replacing NAN'
    csvmap['train'].replace( np.nan, -999999, regex=True, inplace=True)    
    if(predict):
        csvmap['test'].replace( np.nan, -999999, regex=True, inplace=True) 
    #print csvmap['train'].dtypes
    print 'Fitting train trees'  
    train_names = csvmap['train'].columns.values.tolist()
    types = csvmap['train'].dtypes

    test_names = 0
    types_test = 0
    if(predict):
        test_names = csvmap['test'].columns.values.tolist()
        types_test = csvmap['test'].dtypes      
    dates = []
    i = 0
    for item in types:
        if(item == 'object'):
            dates.append(i)
        i = i + 1
    for item in dates:
        lbl =  preprocessing.LabelEncoder()
        lbl.fit( list( csvmap['train'][train_names[item]] ) )
        csvmap['train'][train_names[item]] = lbl.transform( csvmap['train'][train_names[item]] )
    print 'Fitted test tree'
    idtotag = 0     
    if(predict):
        idtotag = csvmap['test']['Id']        
        dates = []
        i=0
        for item in types_test:
    #        #print item
            if(item == 'object'):
                dates.append(i)
            i = i + 1
        for item in dates:
            lbl =  preprocessing.LabelEncoder()
            lbl.fit( list( csvmap['test'][test_names[item]] ) )
            csvmap['test'][test_names[item]] = lbl.transform( csvmap['test'][test_names[item]] )
            
    for item in train_names:
        print item            
        plt.scatter(csvmap['train'][item], csvmap['train']['Sales'] , marker="o")
        plt.xlabel(item, fontsize=18)
        plt.ylabel('Sales', fontsize=16)
        ax = plt.gca()
        ax.set_axis_bgcolor('white')
        plt.savefig(item + '_vs_sales.png')            
            
    print 'Preparing XGBoost'        
    train = csvmap['train']
    tags = csvmap['train']['Sales']
    totag = 0
    if(predict):
        totag = csvmap['test'] 
    params, plst = SettingsForXGBoost( 0.05, 0.03, 25, 20, 0.9 )
    print 'Selecting random rows for training'
    num_rounds = 200
    rounds  = 40
    rows = random.sample( train.index, int(len(tags)*0.99))
    df_train = train.ix[rows]
    df_test = train.drop(rows)
    ##ids_train = df_train['ID']
    
    tags_train = np.log(df_train['Sales'] + 1)  
    df_train = df_train.drop(['Sales'], axis = 1)
    ##ids_test = df_test['ID']
    tags_test = np.log(df_test['Sales'] + 1)
    df_test = df_test.drop(['Sales'], axis = 1)
    df_train = np.array(df_train)
    tags_train = np.array(tags_train)
    df_test = np.array(df_test)
    tags_test = np.array(tags_test)
    if(predict):
        totag = np.array(totag)
        idtotag = np.array(idtotag)
    print 'Creating Matrix'    
    xgtrain = xgb.DMatrix(df_train, label=tags_train, missing = np.nan)
    xgcontr = xgb.DMatrix(df_test, missing = np.nan)
    xtotag = 0
    if(predict):
        xtotag = xgb.DMatrix(totag, missing = np.nan)
    print 'Training'    
    model = xgb.train(plst, xgtrain, num_rounds)
    print 'Predicting'
    preds = model.predict(xgcontr) 
    tagging = 0
    if(predict):
        tagging = model.predict(xtotag)

    #preds[preds > 0.5] = 1
    #preds[preds <= 0.5] = 0
    #if(predict):
    #    tagging[tagging > 0.5] = 1
    #    tagging[tagging <= 0.5] = 0        
    sub = np.subtract( tags_test, preds )
    sub = np.square( sub )
    sumerr = np.sum(sub)
    print 'Error on prediction ' , math.sqrt(sumerr)
    plt.scatter(tags_test, preds , marker="o")
    plt.xlabel('Real Cost', fontsize=18)
    plt.ylabel('Predicted cost', fontsize=16)
    ax = plt.gca()
    ax.set_axis_bgcolor('white')
    plt.savefig('scatterplot_bp.png')
    preds = np.exp(preds) - 1    
    if(predict):
        tagging = np.exp(tagging) - 1
        outpreds = pd.DataFrame({"Id": idtotag, "Sales": tagging})
        outpreds = outpreds.sort_index( by = ["Id"], ascending = [True])
        outpreds.to_csv('benchmark.csv', index=False)  
    ##print tags_test, preds
    print 'It took ' , datetime.now() - startTime
