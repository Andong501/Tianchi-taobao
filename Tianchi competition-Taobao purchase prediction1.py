#Tianchi competition: Taobao purchase prediction version 1

import matplotlib as mlt
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
import numpy as np 
import pandas as pd 
import datetime
from patsy import dmatrices
from sklearn.linear_model import LogisticRegression as lr
from sklearn import svm
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.ensemble import BaggingClassifier
from __future__ import division #精确除法

#import csv
df1 = pd.read_csv('tianchi_fresh_comp_train_user.csv');
df2 = pd.read_csv('tianchi_fresh_comp_train_item.csv');

#build the Feature X
#df_ft = df_Feature for X
df_ft1 = df1.loc[:, ['user_id', 'item_id', 'behavior_type', 'time']];
df_ft1.time = pd.to_datetime(df_ft1.time).values.astype('datetime64[D]');
df_ft2 = df2.loc[:, ['item_id']];
df_ft_clean = df_ft2.join(df_ft1.set_index('item_id'), on='item_id'); #data clean
s_click = (df_ft_clean.behavior_type==1);
s_collec = (df_ft_clean.behavior_type==2);
s_cart = (df_ft_clean.behavior_type==3);
s_buy = (df_ft_clean.behavior_type==4);
d1 = {'click':s_click, 'collection':s_collec, 'cart':s_cart, 'buy':s_buy};
df_add1 = pd.DataFrame(d1);
df_ft = pd.concat([df_ft_clean, df_add1], axis=1);
df_ft.drop(['behavior_type'], axis=1, inplace=True);
df_ft = df_ft.groupby(['user_id', 'item_id', 'time']).sum();

#df_y for y
df_y = df_ft_clean;
d2 = {'buy':s_buy};
df_add2 = pd.DataFrame(d2);
df_y = pd.concat([df_y, df_add2], axis=1);
one_day = np.timedelta64(1,'D');
df_y.time = df_y.time.sub(one_day);
df_y.drop(['behavior_type'], axis=1, inplace=True);
df_y = df_y.groupby(['user_id', 'item_id', 'time']).sum();
df_y.buy = (df_y.buy!=0);
df_y.buy = df_y.buy.astype(int);

df_Xy = df_ft.join(df_y, rsuffix='_next_day');
df_Xy.loc[:, 'buy_next_day'].fillna(0, inplace=True);

#divide train/cv/lacal test/test dataset
idx = pd.IndexSlice;
train_list=[];
for i in range(25, 28):
	train_date = np.datetime64('2014-11-18') + np.timedelta64(i, 'D');
	train_list.append(train_date);
df_Xy_train = df_Xy.loc[idx[:, :, train_list], :];
df_Xy_cv = df_Xy.loc[idx[:, :, [np.datetime64('2014-12-16')]], :];
df_Xy_local_test = df_Xy.loc[idx[:, :, [np.datetime64('2014-12-17')]], :];
df_Xy_test = df_Xy.loc[idx[:, :, [np.datetime64('2014-12-18')]], :];

#calculate F1-score
def F1(y, y_prediction):
	true_pos = ((y_prediction==1.0) & (y==y_prediction)).sum();
	false_pos = ((y_prediction==1.0) & (y!=y_prediction)).sum();
	false_neg = ((y_prediction==0.0) & (y!=y_prediction)).sum();
	precision = true_pos / (true_pos + false_pos);
	recall = true_pos / (true_pos + false_neg);
	f1 = 2 * precision * recall / (precision + recall);
	return f1, precision, recall;

#train the model with sklearn.LR
formula = 'buy_next_day ~ click + collection + cart + buy';
y_train, X_train = dmatrices(formula, data=df_Xy_train, return_type='dataframe');
model = lr(class_weight='balanced');
res = model.fit(X_train, y_train.values.ravel());
model.coef_
#train prediction
train_prediction = model.predict(X_train);
train_accuracy = np.mean(train_prediction==y_train.buy_next_day.values);
train_f1, train_precision, train_recall = F1(y_train.buy_next_day.values, train_prediction);
#cv prediction
y_cv, X_cv = dmatrices(formula, data=df_Xy_cv, return_type='dataframe');
cv_prediction = model.predict(X_cv);
cv_accuracy = np.mean(cv_prediction==y_cv.buy_next_day.values);
cv_f1, cv_precision, cv_recall = F1(y_cv.buy_next_day.values, cv_prediction);
#local test prediction
y_local_test, X_local_test = dmatrices(formula, data=df_Xy_local_test, return_type='dataframe');
local_test_prediction = model.predict(X_local_test);
local_test_accuracy = np.mean(local_test_prediction==y_local_test.buy_next_day.values);
local_test_f1, local_test_precision, local_test_recall = F1(y_local_test.buy_next_day.values, local_test_prediction);
#test prediction submit
y_test, X_test = dmatrices(formula, data=df_Xy_test, return_type='dataframe');
test_prediction = model.predict(X_test);
#build a submit table format
submit_table = y_test;
submit_table.buy_next_day = test_prediction;
submit_table = submit_table.loc[submit_table.buy_next_day==1.0];
submit_pair = submit_table.reset_index(level=[0, 1, 2]);
submit_pair = submit_pair.loc[:, ['user_id', 'item_id']];
submit_pair.user_id = submit_pair.user_id.apply(str);
submit_pair.item_id = submit_pair.item_id.apply(str);
submit_pair.to_csv('tianchi_mobile_recommendation_predict.csv', index=False);


#train the model with linear svm
n_estimators = 10;
model_svm = BaggingClassifier(LinearSVC(class_weight='balanced'), max_samples=1.0 / n_estimators, n_estimators=n_estimators);
res_svm = model_svm.fit(X_train, y_train.values.ravel());
#train prediction
train_prediction_svm = model_svm.predict(X_train);
train_accuracy_svm = np.mean(train_prediction_svm==y_train.buy_next_day.values);
train_f1_svm, train_precision_svm, train_recall_svm = F1(y_train.buy_next_day.values, train_prediction_svm);
#cv prediction
cv_prediction_svm = model_svm.predict(X_cv);
cv_accuracy_svm = np.mean(cv_prediction_svm==y_cv.buy_next_day.values);
cv_f1_svm, cv_precision_svm, cv_recall_svm = F1(y_cv.buy_next_day.values, cv_prediction_svm);
#local test prediction
local_test_prediction_svm = model_svm.predict(X_local_test);
local_test_accuracy_svm = np.mean(local_test_prediction_svm==y_local_test.buy_next_day.values);
local_test_f1_svm, local_test_precision_svm, local_test_recall_svm = F1(y_local_test.buy_next_day.values, local_test_prediction_svm);
#test prediction submit
test_prediction_svm = model.predict(X_test);


#train the model with one-class svm (not get good result!)
X_train_OCS1 = df_Xy_train.loc[df_Xy_train.buy_next_day==0].iloc[:, [0, 1, 2, 3]]; #non-polluted train data
X_train_OCS2 = df_Xy_train.iloc[:, [0, 1, 2, 3]];
y_train_OCS = df_Xy_train.iloc[:, [4]];
X_cv_OCS = df_Xy_cv.iloc[:, [0, 1, 2, 3]];
y_cv_OCS = df_Xy_cv.iloc[:, [4]];
X_local_test_OCS = df_Xy_local_test.iloc[:, [0, 1, 2, 3]];
y_local_test_OCS = df_Xy_local_test.iloc[:, [4]];
X_test_OCS = df_Xy_test.iloc[:, [0, 1, 2, 3]];
y_test_OCS = df_Xy_test.iloc[:, [4]];
model_OCS = svm.OneClassSVM(kernel='rbf', nu=0.0036, gamma=0.1);
res_OCS = model_OCS.fit(X_train_OCS1);
#train prediction
train_prediction_OCS = (model_OCS.predict(X_train_OCS2)==-1).astype(int);
train_accuracy_OCS = np.mean(train_prediction_OCS==y_train_OCS.buy_next_day.values);
train_f1_OCS, train_precision_OCS, train_recall_OCS = F1(y_train_OCS.buy_next_day.values, train_prediction_OCS);
#cv prediction
cv_prediction_OCS = (model_OCS.predict(X_cv_OCS)==-1).astype(int);
cv_accuracy_OCS = np.mean(cv_prediction_OCS==y_cv_OCS.buy_next_day.values);
cv_f1_OCS, cv_precision_OCS, cv_recall_OCS = F1(y_cv_OCS.buy_next_day.values, cv_prediction_OCS);
#train prediction
local_test_prediction_OCS = (model_OCS.predict(X_local_test_OCS)==-1).astype(int);
local_test_accuracy_OCS = np.mean(local_test_prediction_OCS==y_local_test_OCS.buy_next_day.values);
local_test_f1_OCS, local_test_precision_OCS, local_test_recall_OCS = F1(y_local_test_OCS.buy_next_day.values, local_test_prediction_OCS);
#cv prediction
test_prediction_OCS = (model_OCS.predict(X_test_OCS)==-1).astype(int);





