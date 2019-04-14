#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 10 20:11:12 2019

@author: lego
"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns
from scipy.stats import *
import scipy.stats
import theano
import keras

import sklearn


bld_own=pd.read_csv('Building_Ownership_Use.csv')
bld_str=pd.read_csv('Building_Structure.csv')
train=pd.read_csv('train.csv')
test=pd.read_csv('test.csv')
sample=pd.read_csv('sample_submission.csv')

test_id=test['building_id'].copy()

res=pd.merge(bld_own,bld_str,on=['building_id','district_id','vdcmun_id','ward_id'])
res_train=pd.merge(res,train,on='building_id')
res_test=pd.merge(res,test,on='building_id')

rs=res_train
columns=[]
columns=list(res_train.columns.values)

cor=res_train.corr()
sns.heatmap(cor)

un_col=['building_id','district_id_x','vdcmun_id_x']
res_train=res_train.drop(un_col,1)
res_test=res_test.drop(un_col,1)

#RENAME cols
res_train=res_train.rename(columns={'vdcmun_id_y':'vdcmun_id','district_id_y':'district_id'})
res_test=res_test.rename(columns={'vdcmun_id_y':'vdcmun_id','district_id_y':'district_id'})

columns=list(res_train.columns.values)

#GET THE TARGET VARIABLE..
y=res_train['damage_grade'].copy()
res_train=res_train.drop(['damage_grade'],1)

#NOW REMOVE NANS AND MISSING VALUES AND ENCODE CAT VARIABLES..
cat_fea=res_train.dtypes.loc[res_train.dtypes=='object'].index

#dummies for categorical
res_train=pd.get_dummies(data=res_train,columns=cat_fea)
res_train.dtypes

res_test=pd.get_dummies(data=res_test,columns=cat_fea)

#missing valuesss
for i in res_train.columns.values:
    r=res_train[i].mode()
    res_train[i].fillna(r[0],inplace=True)

for i in res_test.columns.values:
    r=res_test[i].mode()
    res_test[i].fillna(r[0],inplace=True)
    
#check if missing still..
isnull=res_train.isnull().sum()
#nONE THERE..PROCEED    

cor1=res_train.corr()
sns.heatmap(cor1)
    
#SCALE VALUES
from sklearn.preprocessing import normalize
res_train=normalize(res_train)
res_test=normalize(res_test)

#NOW THAT THE COLUMNS ARE PERFECT..WE NEED TO DECIDE WHICH COLUMNS(FEATURES) ARE IMPORTANT..
# 1 WAY...MANUAL CHECK FROM CORRMAT
#2 PCA
#3 RANDOM FOREST RANK ARGSORT

#1 min
#im using a non=linear pca...to make sure cat variables are not affected as co
from sklearn.decomposition import PCA
pca=PCA(n_components=5,random_state=100)
trn_chk_pc=pca.fit_transform(res_train)

trn_chk_pc[0:2]
pca.explained_variance_ratio_

#2min
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as lda
lda=lda()
trn_chk=lda.fit_transform(res_train,y)

lda.decision_function
lda.coef_
lda.classes_

lda.explained_variance_ratio_

#SPLIT DATA
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(trn_chk,y,test_size=0.3)

#MODEL 1: RANDOMFOR
#5 min
from sklearn.ensemble import RandomForestClassifier
rf=RandomForestClassifier(n_estimators=600,max_features='auto')
rf.fit(x_train,y_train)

y_pred=rf.predict(x_test)

from sklearn.model_selection import GridSearchCV
parameters=[{'n_estimators':[600]}]
#For integer/None inputs, if the estimator is a classifier and y is either....
#binary or multiclass, StratifiedKFold is used. In all other cases, KFold is used.
gs=GridSearchCV(estimator=rf,param_grid=parameters,scoring='accuracy',cv=3,n_jobs=-1)
gs.fit(x_train,y_train)
b_est=gs.best_estimator_
b_score=gs.best_score_
b_para=gs.best_params_

#no need if kfold done
from sklearn.model_selection import cross_val_score 
acc=cross_val_score(rf,x_test,y_test,cv=3)
print(acc)
#70.01%

#model2: xgboost on pca data / lda data 

import xgboost
from xgboost.sklearn import XGBClassifier
xgb=XGBClassifier(n_estimators=200,objective='multi:softmax',learning_rate=0.001)
xgb.fit(x_train,y_train)
xgb.base_score
xgb.feature_importances_



from sklearn.model_selection import cross_val_score 
acc_xgb=cross_val_score(xgb,x_test,y_test,cv=5)
np.mean(acc_xgb)

#0.70188= 71%

#model 3 : neural network on orig data/ pca data/lda
#need to change target column to 5 columns..

yt=pd.get_dummies(y_train)
yt=np.array(yt)

yte=pd.get_dummies(y_test)
yte=np.array(yte)


import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Activation
from keras.callbacks import History
his=History()
from keras import regularizers
#input layer
cl=Sequential()

#hidden1
cl.add(Dense(output_dim=700,activation='relu',init='uniform',input_shape=x_train.shape[1:],kernel_regularizer=regularizers.l2(0.0001)))
cl.add(Dropout(0.20))
#hidden2
cl.add(Dense(output_dim=500,activation='relu',init='uniform',kernel_regularizer=regularizers.l2(0.0001)))
cl.add(Dropout(0.20))

cl.add(Dense(output_dim=500,activation='relu',init='uniform',kernel_regularizer=regularizers.l2(0.0001)))
cl.add(Dropout(0.20))


cl.add(Dense(output_dim=500,activation='relu',init='uniform',kernel_regularizer=regularizers.l2(0.0001)))
cl.add(Dropout(0.20))


cl.add(Dense(output_dim=100,activation='relu',init='uniform',kernel_regularizer=regularizers.l2(0.0001)))
#output layer
cl.add(Dense(output_dim=5,init='uniform',activation='softmax'))

cl.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])

cl.summary()

cl.fit(x_train,yt,epochs=20,shuffle=True,batch_size=128,validation_data=(x_test,yte),callbacks=[his])

plt.plot(his.history['loss'])

y_pred_nn=cl.predict(x_test)

#70.9=71%


#MODEL 3l: CLUSTERING ALGORITHM
from sklearn.cluster import KMeans

#predict clusters no.
wcss= []
#now we assume we can have a maximum of 11 clusters and then from the wcss function( also refer the elbow method graph) to infer that 
#first 5-6 are optimal number of clusters. :)
for i in range(1,11):
    kmeans=KMeans(n_clusters=i,init='k-means++',random_state=0)
    kmeans.fit(x_train)
    wcss.append(kmeans.inertia_)#here for loop is inside the spaced lines ...so plt doesnt come under for loop..SPACING IS IMPORTANT

plt.plot(range(1,11),wcss)
plt.title('The elbow method')
plt.xlabel('number of clusters (k)')
plt.ylabel('wcss value')
plt.show()

#fitt

kmeans=KMeans(n_clusters=5,init='k-means++',random_state=0)
y_pred_km=kmeans.fit_predict(x_train)

y_kmeans_pred=y_pred_km
np.unique(y_pred_km)#--0,1,2,3,4

plt.scatter(x_train[y_kmeans_pred==0,0],x_train[y_kmeans_pred==0,1],s=100,c='red',label='careful')

plt.scatter(x_train[y_kmeans_pred==1,0],x_train[y_kmeans_pred==1,1],s=100,c='blue',label='standard')

plt.scatter(x_train[y_kmeans_pred==2,0],x_train[y_kmeans_pred==2,1],s=100,c='green',label='target')

plt.scatter(x_train[y_kmeans_pred==3,0],x_train[y_kmeans_pred==3,1],s=100,c='cyan',label='careless')

plt.scatter([y_kmeans_pred==4,0],x_train[y_kmeans_pred==4,1],s=100,c='pink',label='sensible')
plt.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1],s=100,c='yellow',label='cluster centroid')
#THIS WAS FOR CLUSTER CENTERS..SEE Y FOR ALL 5 CLUSTER CENTER COORDINATES
plt.title('ANNUAL INC VS SPENDING SCORE')
plt.xlabel('annual inc')
plt.ylabel('spending score')
plt.legend()
plt.show() 

#FAIL...

#MODEL 4: SVM
#

from sklearn.svm import LinearSVC
svc=LinearSVC(C=0.1)
svc.fit(x_train,y_train)
svc.score

y_pred_svm=svc.predict(x_test)
np.unique(y_pred_svm)

acc_svc=cross_val_score(svc,x_test,y_test,cv=50)
np.mean(acc_svc)
#70.4%4

          
          