# -*- coding: utf-8 -*-
"""
Created on Wed Aug  8 11:38:27 2018

@author: PAVEETHRAN
"""

#OM
#lsvm-0.46450
#xg1=0.50193
#xg2=0.50165


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns
from scipy.stats import norm

bld_own=pd.read_csv('Building_Ownership_Use.csv')
bld_str=pd.read_csv('Building_Structure.csv')
train=pd.read_csv('train.csv')
test=pd.read_csv('test.csv')
sample=pd.read_csv('sample_submission.csv')

"""

a=train.loc[train['building_id']=='a3380c4fd9']

train.size[0]
test.shape[-1]
#TO compare if the values of one table is in other table
for i in range(int(bld_own.shape[0])):
    ops=train[train.building_id.isin([bld_own.building_id[i]])]
            
s=test[test.building_id.isin([bld_own.building_id[0]])]    

#NAIVE APPROACH..
"""

#CONCAT REMOVES DUPLICATES AND EDITS...
#result=pd.concat([bld_own,bld_str],ignore_index=True)

res=pd.merge(bld_own,bld_str,on='building_id')
res_train=pd.merge(res,train,on='building_id')
res_test=pd.merge(res,test,on='building_id')


#or
drop=['building_id']
res_train=res_train.drop(drop,1)
indx=res_test['building_id']
res_test=res_test.drop(drop,1)


#NORMALIZATION OF OUTPUT VARIABLE
res_train.columns.get_values()[1]
res_train.column_index['damage_grade']
y_train=train['damage_grade']

res_train['damage_grade']

sns.distplot(res_train['damage_grade'],fit=norm)

#pos=[5,6,7,8,9,10,12,13,14,15]
#drop=res_train.columns[pos]
#res_train=res_train.drop(drop,1)
#
#pos=[5]
#drop=res_train.columns[pos]
#res_train=res_train.drop(drop,1)
#
#pos=[21,22,23,24,25,26,27,28,29,30,31]
#drop=res_train.columns[pos]
#res_train=res_train.drop(drop,1)
#pos=[0,1,2,5,6,7]
#drop=res_train.columns[pos]
#res_train=res_train.drop(drop,1)
#pos=[0,1,5,27]
#drop=res_train.columns[pos]
#res_train=res_train.drop(drop,1)
#


#test data
res_train=res_train.drop('building_id',1)
res_test=res_test.drop('building_id',1)

#x=res_train[:,57].values
#y_train=otpt[:].values

#TRAINING DATASET....
cat_fea=res_train.dtypes.loc[res_train.dtypes=='object'].index
float_fea=res_train.dtypes.loc[res_train.dtypes=='float'].index
int_fea=res_train.dtypes.loc[res_train.dtypes=='int64'].index

#mising values
res_train[cat_fea].apply(lambda x:len(x.unique()))
res_train[float_fea].apply(lambda x:len(x.unique()))
res_train[int_fea].apply(lambda x:len(x.unique()))



#ONLY FOR A COLUMN THAT CONTAINS A NOT APPLICABLE NAME...CHANGE IT TO MODE ..
#tor=res_train['other_floor_type']
#for a in tor:
#    if(a=='Not applicable'):
#        res_train['other_floor_type']='TImber/Bamboo-Mud'
#

#REMEMBER..FOR CAT VARIABLES...MISSING MAY DENOTE ANOTHER CLASS TOO...SO TWO WAYS..
#EITHER MAKE A NEW TYPE AS MISSING TYPE OR MODE OF THE AVAILABLE DATA.
for i in cat_fea:
    r=res_train[i].mode()
    res_train[i].fillna(r[0],inplace=True)


r=0
for i in float_fea:
    r=res_train[i].mode()
    res_train[i].fillna(r[0],inplace=True)

r=0
for i in int_fea:
    r=res_train[i].mode()
    res_train[i].fillna(r[0],inplace=True)


#uni=np.unique(pr)
#pr.__contains__.__class__

#los=res_train[['legal_ownership_status', 'land_surface_condition', 'foundation_type','roof_type', 'ground_floor_type', 'other_floor_type', 'position',
#'plan_configuration', 'condition_post_eq', 'area_assesed']].copy()
#los=pd.DataFrame(los)
#los.isnull().sum()
#los=los.values

#categorical data
#cat_fea....
#LABEL ENCODING THE CATEGORICAL VALUES
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
le=LabelEncoder()
ohe=OneHotEncoder()
    
    
for var in cat_fea:
    print (var)
    res_train[var]=le.fit_transform(res_train[var])
 

for var in cat_fea:    
    inx=res_train.columns.get_loc(var)
    print (inx)
    ohe=OneHotEncoder(categorical_features=[inx])
    x=ohe.fit_transform(res_train).toarray()
            
#ta=x
#ta=pd.DataFrame(ta)    

#Feature Scaling
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
x=sc.fit_transform(x)

#TEST DATA....

sp=res_test.dtypes[res_test.dtypes=='object'].index
cat_fea=res_train.dtypes.loc[res_test.dtypes=='object'].index
float_fea=res_test.dtypes.loc[res_test.dtypes=='float'].index
int_fea=res_test.dtypes.loc[res_test.dtypes=='int64'].index

#mising values
res_test[cat_fea].apply(lambda x:len(x.unique()))
res_test[float_fea].apply(lambda x:len(x.unique()))
res_test[int_fea].apply(lambda x:len(x.unique()))

#REMEMBER..FOR CAT VARIABLES...MISSING MAY DENOTE ANOTHER CLASS TOO...SO TWO WAYS..
#EITHER MAKE A NEW TYPE AS MISSING TYPE OR MODE OF THE AVAILABLE DATA.
for i in cat_fea:
    r=res_test[i].mode()
    res_test[i].fillna(r[0],inplace=True)

r=0
for i in float_fea:
    r=res_test[i].mode()
    res_test[i].fillna(r[0],inplace=True)

r=0
for i in int_fea:
    r=res_test[i].mode()
    res_test[i].fillna(r[0],inplace=True)


#uni=np.unique(pr)
#pr.__contains__.__class__

#los=res_test[['legal_ownership_status', 'land_surface_condition', 'foundation_type','roof_type', 'ground_floor_type', 'other_floor_type', 'position',
#'plan_configuration', 'condition_post_eq', 'area_assesed']].copy()
#los=pd.DataFrame(los)
#los.isnull().sum()
#los=los.values

#categorical data
#cat_fea....
#LABEL ENCODING THE CATEGORICAL VALUES
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
le=LabelEncoder()
ohe=OneHotEncoder()
    
    
for var in cat_fea:
    print (var)
    res_test[var]=le.fit_transform(res_test[var])
 
#FOR VALIDATA
for var in cat_fea:    
    inx=res_test.columns.get_loc(var)
    print (inx)
    ohe=OneHotEncoder(categorical_features=[inx])
    validata=ohe.fit_transform(res_test).toarray()
            

otpt=le.fit_transform(otpt)
np.unique(otpt)
    
#Feature Scaling
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
validata=sc.fit_transform(validata)

#integ=x.astype('int')

#SPLIT DATA

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,otpt,test_size=0.30,random_state=0)


# Applying LDA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
lda = LDA()
x_train = lda.fit_transform(x_train, y_train)
x_test = lda.transform(x_test)
validata=lda.transform(validata)
rat=lda.explained_variance_ratio_


#PCA APPLY...
from sklearn.decomposition import PCA
pca=PCA(n_components=19)
x_train=pca.fit_transform(x_train)
x_test=pca.transform(x_test)
validata=pca.transform(validata)
exp_variance_ratio=pca.explained_variance_ratio_
exp_variance=pca.explained_variance_
mean=pca.components_

#KPCAS
from sklearn.decomposition import KernelPCA
kpca = KernelPCA(n_components = 10, kernel = 'rbf')
x_train = kpca.fit_transform(x_train)
x_test = kpca.transform(x_test)


# Fitting Logistic Regression to the Training set
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
classifier.fit(x_train, y_train)
y_prediction=classifier.predict(validata)



#KFOLD
from sklearn.cross_validation import cross_val_score as cvs
acc=[]
acc=cvs(estimator=cl,X=x_train,y=y_train,cv=5,n_jobs=-1)  
np.mean(acc)



#do with linearsvc..it has by defaul one vs rest strategy..where as normal svc has one vs one only....
#fitting svc
from sklearn.svm import LinearSVC
cl=LinearSVC()
cl.fit(x_train,y_train)

y_pred=cl.predict(x_test)
np.unique(y_pred)

y_pred=cl.predict(validata)

from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_pred)


#model selection and grid search(para tuning)
from sklearn.model_selection import GridSearchCV
parameters=[{'C':[1,100]}]
#For integer/None inputs, if the estimator is a classifier and y is either....
#binary or multiclass, StratifiedKFold is used. In all other cases, KFold is used.
gs=GridSearchCV(estimator=cl,param_grid=parameters,scoring='accuracy',cv=3,n_jobs=-1)
gs.fit(x_train,y_train)


b_est=gs.best_estimator_
b_score=gs.best_score_
b_para=gs.best_params_


#xgboost
import xgboost
from xgboost.sklearn import XGBClassifier
#CHECK THE ML ,=MASTERY SITE FOR XGBOOST LEARNING RATE AND N ESTIMATORS SUMMARY....
xgb=XGBClassifier(n_estimators=350,objective='multi:softmax',learning_rate=0.2)
xgb.fit(x_train,y_train)
y_prob=xgb.predict_proba(x_test)
y_p=xgb.predict(x_test)
y_pred_xg=xgb.predict(validata)



tp=y_pred
y_pred=[]
for i in tp:
    if(i==0):
       y_pred.append('Grade 1')
    elif(i==1):
        y_pred.append('Grade 2')
    elif(i==2):
        y_pred.append('Grade 3')
    elif(i==3):
        y_pred.append('Grade 4')
    elif(i==4):
        y_pred.append('Grade 5')
        
y_pred=pd.DataFrame(y_pred)
y_pred=y_pred.values
my_list=map(lambda x: x[0],y_pred)
y_pred=pd.Series(my_list)

subs=pd.DataFrame({'building_id': ind,'damage_grade':y_pred})
subs=pd.DataFrame(subs)
subs.to_csv('sol_pred.csv',index=False)


subs=pd.read_csv('sol_pred.csv')

sample=sample.drop('damage_grade',1)
y=subs['damage_grade'].copy()
sample['damage_grade']=pd.Series(y)
sol=sample
sol.to_csv('sol_pred1.csv',index=False)


#
#y_pred.unique
#subs.isnull().sum()
#subs.loc[subs['building_id']=='25ec24ced18']
#ANN
#import keras
#from keras.models import Sequential
#from keras.layers import Dense
#from keras.layers import Dropout
#from keras.layers import Activation
#
#import tensorflow as tf
#    
##input layer
#cl=Sequential()
#
##hidden1
#cl.add(Dense(output_dim=6,activation='relu',init='uniform',input_dim=60))
#cl.add(Dropout(0.5))
##hidden2
#cl.add(Dense(output_dim=6,activation='relu',init='uniform'))
#cl.add(Dropout(0.5))
##output layer
#cl.add(Dense(output_dim=5,init='uniform'))
#cl.add(Activation(tf.nn.softmax))
#
#cl.compile(optimizer='rmsprop',loss='binary_crossentropy',metrics=['accuracy'])
#
#cl.fit(x_train,y_train,epochs=5,batch_size=100000)
#
##prediction....
#y_pred_nn=cl.predict(x_test)
#np.unique(y_pred_nn)

