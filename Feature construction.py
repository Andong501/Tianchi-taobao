##Tianchi competition: Taobao purchase prediction feature construction

import matplotlib as mlt
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
import numpy as np 
import pandas as pd 
import datetime
import warnings
from patsy import dmatrices
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import metrics
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression as lr
from sklearn import svm
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.ensemble import BaggingClassifier
from __future__ import division #精确除法

##import csv
df1 = pd.read_csv('tianchi_fresh_comp_train_user.csv');
df2 = pd.read_csv('tianchi_fresh_comp_train_item.csv');

##data clean
df2.drop(['item_geohash'], axis=1, inplace=True);
df2.drop_duplicates(inplace=True);
df1.drop(['user_geohash'], axis=1, inplace=True);
df1.time = pd.to_datetime(df1.time).values.astype('datetime64[D]');
df_clean = df1.join(df2.drop(['item_category'], axis=1, inplace=False).set_index('item_id'), on='item_id', how='right');

##feature extraction
##for user
df_X_u1 = df_clean.loc[df_clean.behavior_type==4].groupby('user_id').size().reset_index(name='ur_buy').set_index('user_id');
df_X_u2 = df_clean.loc[df_clean.behavior_type==1].groupby('user_id').size().reset_index(name='ur_click').set_index('user_id');
df_X_u3 = df_clean.loc[df_clean.behavior_type==2].groupby('user_id').size().reset_index(name='ur_collec').set_index('user_id');
df_X_u4 = df_clean.loc[df_clean.behavior_type==3].groupby('user_id').size().reset_index(name='ur_cart').set_index('user_id');

df_X_u5 = df_X_u1.rename(columns={'ur_buy':'ur_bh_cvr'});
df_X_u5 = df_X_u5.join(df_X_u1, how='outer').join(df_X_u2, how='outer').join(df_X_u3, how='outer').join(df_X_u4, how='outer');
df_X_u5.fillna(0, inplace=True);
df_X_u5.ur_bh_cvr = df_X_u5.ur_buy / (df_X_u5.ur_buy + df_X_u5.ur_click + df_X_u5.ur_collec + df_X_u5.ur_cart);
df_X_u5.drop(['ur_buy', 'ur_click', 'ur_collec', 'ur_cart'], axis=1, inplace=True);

df_X_u6 = df_X_u1.rename(columns={'ur_buy':'ur_ck_cvr'});
df_X_u6 = df_X_u6.join(df_X_u1, how='outer').join(df_X_u2, how='outer');
df_X_u6.fillna(0, inplace=True);
df_X_u6.ur_ck_cvr = df_X_u6.ur_buy / df_X_u6.ur_click;
df_X_u6.replace(np.inf, 0, inplace=True);
df_X_u6.drop(['ur_buy', 'ur_click'], axis=1, inplace=True);

df_X_u7 = df_X_u1.rename(columns={'ur_buy':'ur_cl_cvr'});
df_X_u7 = df_X_u7.join(df_X_u1, how='outer').join(df_X_u3, how='outer');
df_X_u7.fillna(0, inplace=True);
df_X_u7.ur_cl_cvr = df_X_u7.ur_buy / df_X_u7.ur_collec;
df_X_u7.replace(np.inf, 0, inplace=True);
df_X_u7.drop(['ur_buy', 'ur_collec'], axis=1, inplace=True);

df_X_u8 = df_X_u1.rename(columns={'ur_buy':'ur_ct_cvr'});
df_X_u8 = df_X_u8.join(df_X_u1, how='outer').join(df_X_u4, how='outer');
df_X_u8.fillna(0, inplace=True);
df_X_u8.ur_ct_cvr = df_X_u8.ur_buy / df_X_u8.ur_cart;
df_X_u8.replace(np.inf, 0, inplace=True);
df_X_u8.drop(['ur_buy', 'ur_cart'], axis=1, inplace=True);

df_X_u9 = df_clean.loc[df_clean.behavior_type==4].groupby(['item_id', 'user_id']).size().reset_index(name='ur_buy_dp');
df_X_u9.ur_buy_dp = (df_X_u9.ur_buy_dp!=1);
df_X_u9.drop(['item_id'], axis=1, inplace=True);
df_X_u9 = df_X_u9.groupby('user_id').mean();

df_X_u10 = df_clean.loc[df_clean.behavior_type==4].groupby(['user_id', 'time']).size().reset_index(name='ur_co_buy_pre');
df_X_u10.ur_co_buy_pre = (df_X_u10.ur_co_buy_pre!=0).astype(int);
df_X_u10s = df_clean.loc[df_clean.behavior_type==4];
df_X_u10s.time = df_X_u10s.time.sub(np.timedelta64(1, 'D'));
df_X_u10s = df_X_u10s.groupby(['user_id', 'time']).size().reset_index(name='ur_buy_nextday');
df_X_u10s.ur_buy_nextday = (df_X_u10s.ur_buy_nextday!=0).astype(int);
df_X_u10 = df_X_u10.set_index(['user_id', 'time']).join(df_X_u10s.set_index(['user_id', 'time']), how='outer');
df_X_u10.fillna(0, inplace=True);
df_X_u10 = df_X_u10.astype(bool);
df_X_u10.ur_co_buy_pre = (df_X_u10.ur_co_buy_pre & df_X_u10.ur_buy_nextday);
df_X_u10.ur_co_buy_pre = df_X_u10.ur_co_buy_pre.astype(int);
df_X_u10.drop(['ur_buy_nextday'], axis=1, inplace=True);
df_X_u10.drop([np.datetime64('2014-11-17'), np.datetime64('2014-12-18')], axis=0, level=1, inplace=True);
df_X_u10 = df_X_u10.reset_index(level=[0, 1]);
df_X_u10.drop(['time'], axis=1, inplace=True);
df_X_u10 = df_X_u10.groupby(['user_id']).mean();

##for item(category)
df_X_c1 = df_clean.loc[df_clean.behavior_type==4].groupby('item_category').size().reset_index(name='cg_buy').set_index('item_category');
df_X_c2 = df_clean.loc[df_clean.behavior_type==1].groupby('item_category').size().reset_index(name='cg_click').set_index('item_category');
df_X_c3 = df_clean.loc[df_clean.behavior_type==2].groupby('item_category').size().reset_index(name='cg_collec').set_index('item_category');
df_X_c4 = df_clean.loc[df_clean.behavior_type==3].groupby('item_category').size().reset_index(name='cg_cart').set_index('item_category');

df_X_c5 = df_X_c1.rename(columns={'cg_buy':'cg_bh_cvr'});
df_X_c5 = df_X_c5.join(df_X_c1, how='outer').join(df_X_c2, how='outer').join(df_X_c3, how='outer').join(df_X_c4, how='outer');
df_X_c5.fillna(0, inplace=True);
df_X_c5.cg_bh_cvr = df_X_c5.cg_buy / (df_X_c5.cg_buy + df_X_c5.cg_click + df_X_c5.cg_collec + df_X_c5.cg_cart);
df_X_c5.drop(['cg_buy', 'cg_click', 'cg_collec', 'cg_cart'], axis=1, inplace=True);

df_X_c6 = df_X_c1.rename(columns={'cg_buy':'cg_ck_cvr'});
df_X_c6 = df_X_c6.join(df_X_c1, how='outer').join(df_X_c2, how='outer');
df_X_c6.fillna(0, inplace=True);
df_X_c6.cg_ck_cvr = df_X_c6.cg_buy / df_X_c6.cg_click;
df_X_c6.replace(np.inf, 0, inplace=True);
df_X_c6.drop(['cg_buy', 'cg_click'], axis=1, inplace=True);

df_X_c7 = df_X_c1.rename(columns={'cg_buy':'cg_cl_cvr'});
df_X_c7 = df_X_c7.join(df_X_c1, how='outer').join(df_X_c3, how='outer');
df_X_c7.fillna(0, inplace=True);
df_X_c7.cg_cl_cvr = df_X_c7.cg_buy / df_X_c7.cg_collec;
df_X_c7.replace(np.inf, 0, inplace=True);
df_X_c7.drop(['cg_buy', 'cg_collec'], axis=1, inplace=True);

df_X_c8 = df_X_c1.rename(columns={'cg_buy':'cg_ct_cvr'});
df_X_c8 = df_X_c8.join(df_X_c1, how='outer').join(df_X_c4, how='outer');
df_X_c8.fillna(0, inplace=True);
df_X_c8.cg_ct_cvr = df_X_c8.cg_buy / df_X_c8.cg_cart;
df_X_c8.replace(np.inf, 0, inplace=True);
df_X_c8.drop(['cg_buy', 'cg_cart'], axis=1, inplace=True);

df_X_c9 = df_X_c1.rename(columns={'cg_buy':'cg_buy_pr'});
df_X_c9 = df_X_c9.join(df_X_c1, how='outer');
df_X_c9.cg_buy_pr = df_X_c9.cg_buy / df_clean.loc[df_clean.behavior_type==4].iloc[:, 0].count();
df_X_c9.drop(['cg_buy'], axis=1, inplace=True);

df_X_c10 = df_X_c2.rename(columns={'cg_click':'cg_ck_pr'});
df_X_c10 = df_X_c10.join(df_X_c2, how='outer');
df_X_c10.cg_ck_pr = df_X_c10.cg_click / df_clean.loc[df_clean.behavior_type==1].iloc[:, 0].count();
df_X_c10.drop(['cg_click'], axis=1, inplace=True);

df_X_c11 = df_X_c3.rename(columns={'cg_collec':'cg_cl_pr'});
df_X_c11 = df_X_c11.join(df_X_c3, how='outer');
df_X_c11.cg_cl_pr = df_X_c11.cg_collec / df_clean.loc[df_clean.behavior_type==2].iloc[:, 0].count();
df_X_c11.drop(['cg_collec'], axis=1, inplace=True);

df_X_c12 = df_X_c4.rename(columns={'cg_cart':'cg_ct_pr'});
df_X_c12 = df_X_c12.join(df_X_c4, how='outer');
df_X_c12.cg_ct_pr = df_X_c12.cg_cart / df_clean.loc[df_clean.behavior_type==3].iloc[:, 0].count();
df_X_c12.drop(['cg_cart'], axis=1, inplace=True);

df_X_c13 = df_clean.loc[df_clean.behavior_type==4].groupby(['user_id', 'item_category']).size().reset_index(name='cg_buy_dp');
df_X_c13.cg_buy_dp = (df_X_c13.cg_buy_dp!=1);
df_X_c13.drop(['user_id'], axis=1, inplace=True);
df_X_c13 = df_X_c13.groupby('item_category').mean();

df_X_c14 = df_clean.loc[df_clean.behavior_type==4].groupby(['item_category', 'time']).size().reset_index(name='ct_co_buy_pre');
df_X_c14.ct_co_buy_pre = (df_X_c14.ct_co_buy_pre!=0).astype(int);
df_X_c14s = df_clean.loc[df_clean.behavior_type==4];
df_X_c14s.time = df_X_c14s.time.sub(np.timedelta64(1, 'D'));
df_X_c14s = df_X_c14s.groupby(['item_category', 'time']).size().reset_index(name='ct_buy_nextday');
df_X_c14s.ct_buy_nextday = (df_X_c14s.ct_buy_nextday!=0).astype(int);
df_X_c14 = df_X_c14.set_index(['item_category', 'time']).join(df_X_c14s.set_index(['item_category', 'time']), how='outer');
df_X_c14.fillna(0, inplace=True);
df_X_c14 = df_X_c14.astype(bool);
df_X_c14.ct_co_buy_pre = (df_X_c14.ct_co_buy_pre & df_X_c14.ct_buy_nextday);
df_X_c14.ct_co_buy_pre = df_X_c14.ct_co_buy_pre.astype(int);
df_X_c14.drop(['ct_buy_nextday'], axis=1, inplace=True);
df_X_c14.drop([np.datetime64('2014-11-17'), np.datetime64('2014-12-18')], axis=0, level=1, inplace=True);
df_X_c14 = df_X_c14.reset_index(level=[0, 1]);
df_X_c14.drop(['time'], axis=1, inplace=True);
df_X_c14 = df_X_c14.groupby(['item_category']).mean();

df_X_i1 = df_clean.loc[df_clean.behavior_type==4].groupby('item_id').size().reset_index(name='it_buy').set_index('item_id');
df_X_i2 = df_clean.loc[df_clean.behavior_type==1].groupby('item_id').size().reset_index(name='it_click').set_index('item_id');
df_X_i3 = df_clean.loc[df_clean.behavior_type==2].groupby('item_id').size().reset_index(name='it_collec').set_index('item_id');
df_X_i4 = df_clean.loc[df_clean.behavior_type==3].groupby('item_id').size().reset_index(name='it_cart').set_index('item_id');

df_X_i5 = df_X_i1.rename(columns={'it_buy':'it_bh_cvr'});
df_X_i5 = df_X_i5.join(df_X_i1, how='outer').join(df_X_i2, how='outer').join(df_X_i3, how='outer').join(df_X_i4, how='outer');
df_X_i5.fillna(0, inplace=True);
df_X_i5.it_bh_cvr = df_X_i5.it_buy / (df_X_i5.it_buy + df_X_i5.it_click + df_X_i5.it_collec + df_X_i5.it_cart);
df_X_i5.drop(['it_buy', 'it_click', 'it_collec', 'it_cart'], axis=1, inplace=True);

df_X_i6 = df_X_i1.rename(columns={'it_buy':'it_ck_cvr'});
df_X_i6 = df_X_i6.join(df_X_i1, how='outer').join(df_X_i2, how='outer');
df_X_i6.fillna(0, inplace=True);
df_X_i6.it_ck_cvr = df_X_i6.it_buy / df_X_i6.it_click;
df_X_i6.replace(np.inf, 0, inplace=True);
df_X_i6.drop(['it_buy', 'it_click'], axis=1, inplace=True);

df_X_i7 = df_X_i1.rename(columns={'it_buy':'it_cl_cvr'});
df_X_i7 = df_X_i7.join(df_X_i1, how='outer').join(df_X_i3, how='outer');
df_X_i7.fillna(0, inplace=True);
df_X_i7.it_cl_cvr = df_X_i7.it_buy / df_X_i7.it_collec;
df_X_i7.replace(np.inf, 0, inplace=True);
df_X_i7.drop(['it_buy', 'it_collec'], axis=1, inplace=True);

df_X_i8 = df_X_i1.rename(columns={'it_buy':'it_ct_cvr'});
df_X_i8 = df_X_i8.join(df_X_i1, how='outer').join(df_X_i4, how='outer');
df_X_i8.fillna(0, inplace=True);
df_X_i8.it_ct_cvr = df_X_i8.it_buy / df_X_i8.it_cart;
df_X_i8.replace(np.inf, 0, inplace=True);
df_X_i8.drop(['it_buy', 'it_cart'], axis=1, inplace=True);

df_X_i9 = df_X_i1.rename(columns={'it_buy':'it_buy_pr'});
df_X_i9 = df_X_i9.join(df_X_i1, how='outer').join(df2.set_index('item_id'), how='left').join(df_X_c1, on='item_category', how='outer');
df_X_i9.it_buy_pr = df_X_i9.it_buy / df_X_i9.cg_buy;
df_X_i9.drop(['it_buy', 'item_category', 'cg_buy'], axis=1, inplace=True);

df_X_i10 = df_X_i2.rename(columns={'it_click':'it_ck_pr'});
df_X_i10 = df_X_i10.join(df_X_i2, how='outer').join(df2.set_index('item_id'), how='left').join(df_X_c2, on='item_category', how='outer');
df_X_i10.it_ck_pr = df_X_i10.it_click / df_X_i10.cg_click;
df_X_i10.drop(['it_click', 'item_category', 'cg_click'], axis=1, inplace=True);

df_X_i11 = df_X_i3.rename(columns={'it_collec':'it_cl_pr'});
df_X_i11 = df_X_i11.join(df_X_i3, how='outer').join(df2.set_index('item_id'), how='left').join(df_X_c3, on='item_category', how='outer');
df_X_i11.it_cl_pr = df_X_i11.it_collec / df_X_i11.cg_collec;
df_X_i11.drop(['it_collec', 'item_category', 'cg_collec'], axis=1, inplace=True);

df_X_i12 = df_X_i4.rename(columns={'it_cart':'it_ct_pr'});
df_X_i12 = df_X_i12.join(df_X_i4, how='outer').join(df2.set_index('item_id'), how='left').join(df_X_c4, on='item_category', how='outer');
df_X_i12.it_ct_pr = df_X_i12.it_cart / df_X_i12.cg_cart;
df_X_i12.drop(['it_cart', 'item_category', 'cg_cart'], axis=1, inplace=True);

df_X_i13 = df_clean.loc[df_clean.behavior_type==4].groupby(['user_id', 'item_id']).size().reset_index(name='it_buy_dp');
df_X_i13.it_buy_dp = (df_X_i13.it_buy_dp!=1);
df_X_i13.drop(['user_id'], axis=1, inplace=True);
df_X_i13 = df_X_i13.groupby('item_id').mean();

df_X_i14 = df_clean.loc[df_clean.behavior_type==4].groupby(['item_id', 'time']).size().reset_index(name='it_co_buy_pre');
df_X_i14.it_co_buy_pre = (df_X_i14.it_co_buy_pre!=0).astype(int);
df_X_i14s = df_clean.loc[df_clean.behavior_type==4];
df_X_i14s.time = df_X_i14s.time.sub(np.timedelta64(1, 'D'));
df_X_i14s = df_X_i14s.groupby(['item_id', 'time']).size().reset_index(name='it_buy_nextday');
df_X_i14s.it_buy_nextday = (df_X_i14s.it_buy_nextday!=0).astype(int);
df_X_i14 = df_X_i14.set_index(['item_id', 'time']).join(df_X_i14s.set_index(['item_id', 'time']), how='outer');
df_X_i14.fillna(0, inplace=True);
df_X_i14 = df_X_i14.astype(bool);
df_X_i14.it_co_buy_pre = (df_X_i14.it_co_buy_pre & df_X_i14.it_buy_nextday);
df_X_i14.it_co_buy_pre = df_X_i14.it_co_buy_pre.astype(int);
df_X_i14.drop(['it_buy_nextday'], axis=1, inplace=True);
df_X_i14.drop([np.datetime64('2014-11-17'), np.datetime64('2014-12-18')], axis=0, level=1, inplace=True);
df_X_i14 = df_X_i14.reset_index(level=[0, 1]);
df_X_i14.drop(['time'], axis=1, inplace=True);
df_X_i14 = df_X_i14.groupby(['item_id']).mean();

##for time
tm_rng = pd.date_range('2014-11-18', periods=32, freq='D');
df_X_t1 = pd.DataFrame({'time':tm_rng, 'weekday':pd.Series(np.random.randn(len(tm_rng))), 'isweekend':pd.Series(np.random.randn(len(tm_rng))), 'isfestival':pd.Series(np.random.randn(len(tm_rng))), 'isfestival_nd':pd.Series(np.random.randn(len(tm_rng))),'isfestival_n2d':pd.Series(np.random.randn(len(tm_rng)))});
df_X_t1.weekday = df_X_t1.time.dt.dayofweek;
df_X_t1.isweekend = ((df_X_t1.weekday==6) | (df_X_t1.weekday==0)).astype(int);
df_X_t1.isfestival = 0;
df_X_t1.isfestival.loc[df_X_t1.time==np.datetime64('2014-12-12')] = 1;
df_X_t1.isfestival_nd = 0;
df_X_ti.isfestival_nd.loc[df_X_t1.time==np.datetime64('2014-12-11')] = 1;
df_X_t1.isfestival_n2d = 0;
df_X_ti.isfestival_n2d.loc[df_X_t1.time==np.datetime64('2014-12-10')] = 1;
a = OneHotEncoder(sparse = False).fit_transform(df_X_t1.loc[:, ['weekday']]);
for i in range(0, 7):
	loc = df_X_t1.shape[1];
	column = 'weekday_' + str(i);
	value = pd.Series(a[:, i]);
	df_X_t1.insert(loc=loc, column=column, value=value);
df_X_t1.drop(['weekday'], axis=1, inplace=True);
df_X_t1 = df_X_t1.set_index('time');

##for user X item(category)
df_X_uc1 = df_clean.loc[df_clean.behavior_type==4].groupby(['user_id', 'item_category']).size().reset_index(name='uc_buy').set_index(['user_id', 'item_category']);
df_X_uc2 = df_clean.loc[df_clean.behavior_type==1].groupby(['user_id', 'item_category']).size().reset_index(name='uc_click').set_index(['user_id', 'item_category']);
df_X_uc3 = df_clean.loc[df_clean.behavior_type==2].groupby(['user_id', 'item_category']).size().reset_index(name='uc_collec').set_index(['user_id', 'item_category']);
df_X_uc4 = df_clean.loc[df_clean.behavior_type==3].groupby(['user_id', 'item_category']).size().reset_index(name='uc_cart').set_index(['user_id', 'item_category']);

df_X_uc5 = df_X_uc1.rename(columns={'uc_buy':'uc_bh_cvr'});
df_X_uc5 = df_X_uc5.join(df_X_uc1, how='outer').join(df_X_uc2, how='outer').join(df_X_uc3, how='outer').join(df_X_uc4, how='outer');
df_X_uc5.fillna(0, inplace=True);
df_X_uc5.uc_bh_cvr = df_X_uc5.uc_buy / (df_X_uc5.uc_buy + df_X_uc5.uc_click + df_X_uc5.uc_collec + df_X_uc5.uc_cart);
df_X_uc5.drop(['uc_buy', 'uc_click', 'uc_collec', 'uc_cart'], axis=1, inplace=True);

df_X_uc6 = df_X_uc1.rename(columns={'uc_buy':'uc_ck_cvr'});
df_X_uc6 = df_X_uc6.join(df_X_uc1, how='outer').join(df_X_uc2, how='outer');
df_X_uc6.fillna(0, inplace=True);
df_X_uc6.uc_ck_cvr = df_X_uc6.uc_buy / df_X_uc6.uc_click;
df_X_uc6.replace(np.inf, 0, inplace=True);
df_X_uc6.drop(['uc_buy', 'uc_click'], axis=1, inplace=True);

df_X_uc7 = df_X_uc1.rename(columns={'uc_buy':'uc_cl_cvr'});
df_X_uc7 = df_X_uc7.join(df_X_uc1, how='outer').join(df_X_uc3, how='outer');
df_X_uc7.fillna(0, inplace=True);
df_X_uc7.uc_cl_cvr = df_X_uc7.uc_buy / df_X_uc7.uc_collec;
df_X_uc7.replace(np.inf, 0, inplace=True);
df_X_uc7.drop(['uc_buy', 'uc_collec'], axis=1, inplace=True);

df_X_uc8 = df_X_uc1.rename(columns={'uc_buy':'uc_ct_cvr'});
df_X_uc8 = df_X_uc8.join(df_X_uc1, how='outer').join(df_X_uc4, how='outer');
df_X_uc8.fillna(0, inplace=True);
df_X_uc8.uc_ct_cvr = df_X_uc8.uc_buy / df_X_uc8.uc_cart;
df_X_uc8.replace(np.inf, 0, inplace=True);
df_X_uc8.drop(['uc_buy', 'uc_cart'], axis=1, inplace=True);

df_X_uc9 = df_X_uc1.rename(columns={'uc_buy':'uc_buy_pr'});
df_X_uc9 = df_X_uc9.join(df_X_uc1, how='outer').join(df_X_u1, how='left');
df_X_uc9.uc_buy_pr = df_X_uc9.uc_buy / df_X_uc9.ur_buy;
df_X_uc9.drop(['uc_buy', 'ur_buy'], axis=1, inplace=True);

df_X_uc10 = df_X_uc2.rename(columns={'uc_click':'uc_ck_pr'});
df_X_uc10 = df_X_uc10.join(df_X_uc2, how='outer').join(df_X_u2, how='left');
df_X_uc10.uc_ck_pr = df_X_uc10.uc_click / df_X_uc10.ur_click;
df_X_uc10.drop(['uc_click', 'ur_click'], axis=1, inplace=True);

df_X_uc11 = df_X_uc3.rename(columns={'uc_collec':'uc_cl_pr'});
df_X_uc11 = df_X_uc11.join(df_X_uc3, how='outer').join(df_X_u3, how='left');
df_X_uc11.uc_cl_pr = df_X_uc11.uc_collec / df_X_uc11.ur_collec;
df_X_uc11.drop(['uc_collec', 'ur_collec'], axis=1, inplace=True);

df_X_uc12 = df_X_uc4.rename(columns={'uc_cart':'uc_ct_pr'});
df_X_uc12 = df_X_uc12.join(df_X_uc4, how='outer').join(df_X_u4, how='left');
df_X_uc12.uc_ct_pr = df_X_uc12.uc_cart / df_X_uc12.ur_cart;
df_X_uc12.drop(['uc_cart', 'ur_cart'], axis=1, inplace=True);

df_X_uc13 = df_clean.loc[df_clean.behavior_type==4].groupby(['user_id', 'item_category', 'time']).size().reset_index(name='uc_co_buy_pre');
df_X_uc13.uc_co_buy_pre = (df_X_uc13.uc_co_buy_pre!=0).astype(int);
df_X_uc13s = df_clean.loc[df_clean.behavior_type==4];
df_X_uc13s.time = df_X_uc13s.time.sub(np.timedelta64(1, 'D'));
df_X_uc13s = df_X_uc13s.groupby(['user_id', 'item_category', 'time']).size().reset_index(name='uc_buy_nextday');
df_X_uc13s.uc_buy_nextday = (df_X_uc13s.uc_buy_nextday!=0).astype(int);
df_X_uc13 = df_X_uc13.set_index(['user_id', 'item_category', 'time']).join(df_X_uc13s.set_index(['user_id', 'item_category', 'time']), how='outer');
df_X_uc13.fillna(0, inplace=True);
df_X_uc13 = df_X_uc13.astype(bool);
df_X_uc13.uc_co_buy_pre = (df_X_uc13.uc_co_buy_pre & df_X_uc13.uc_buy_nextday);
df_X_uc13.uc_co_buy_pre = df_X_uc13.uc_co_buy_pre.astype(int);
df_X_uc13.drop(['uc_buy_nextday'], axis=1, inplace=True);
df_X_uc13.drop([np.datetime64('2014-11-17'), np.datetime64('2014-12-18')], axis=0, level=1, inplace=True);
df_X_uc13 = df_X_uc13.reset_index(level=[0, 1, 2]);
df_X_uc13.drop(['time'], axis=1, inplace=True);
df_X_uc13 = df_X_uc13.groupby(['user_id', 'item_category']).mean();

df_X_ui1 = df_clean.loc[df_clean.behavior_type==4].groupby(['user_id', 'item_id']).size().reset_index(name='ui_buy').set_index(['user_id', 'item_id']);
df_X_ui2 = df_clean.loc[df_clean.behavior_type==1].groupby(['user_id', 'item_id']).size().reset_index(name='ui_click').set_index(['user_id', 'item_id']);
df_X_ui3 = df_clean.loc[df_clean.behavior_type==2].groupby(['user_id', 'item_id']).size().reset_index(name='ui_collec').set_index(['user_id', 'item_id']);
df_X_ui4 = df_clean.loc[df_clean.behavior_type==3].groupby(['user_id', 'item_id']).size().reset_index(name='ui_cart').set_index(['user_id', 'item_id']);

df_X_ui5 = df_X_ui1.rename(columns={'ui_buy':'ui_bh_cvr'});
df_X_ui5 = df_X_ui5.join(df_X_ui1, how='outer').join(df_X_ui2, how='outer').join(df_X_ui3, how='outer').join(df_X_ui4, how='outer');
df_X_ui5.fillna(0, inplace=True);
df_X_ui5.ui_bh_cvr = df_X_ui5.ui_buy / (df_X_ui5.ui_buy + df_X_ui5.ui_click + df_X_ui5.ui_collec + df_X_ui5.ui_cart);
df_X_ui5.drop(['ui_buy', 'ui_click', 'ui_collec', 'ui_cart'], axis=1, inplace=True);

df_X_ui6 = df_X_ui1.rename(columns={'ui_buy':'ui_ck_cvr'});
df_X_ui6 = df_X_ui6.join(df_X_ui1, how='outer').join(df_X_ui2, how='outer');
df_X_ui6.fillna(0, inplace=True);
df_X_ui6.ui_ck_cvr = df_X_ui6.ui_buy / df_X_ui6.ui_click;
df_X_ui6.replace(np.inf, 0, inplace=True);
df_X_ui6.drop(['ui_buy', 'ui_click'], axis=1, inplace=True);

df_X_ui7 = df_X_ui1.rename(columns={'ui_buy':'ui_cl_cvr'});
df_X_ui7 = df_X_ui7.join(df_X_ui1, how='outer').join(df_X_ui3, how='outer');
df_X_ui7.fillna(0, inplace=True);
df_X_ui7.ui_cl_cvr = df_X_ui7.ui_buy / df_X_ui7.ui_collec;
df_X_ui7.replace(np.inf, 0, inplace=True);
df_X_ui7.drop(['ui_buy', 'ui_collec'], axis=1, inplace=True);

df_X_ui8 = df_X_ui1.rename(columns={'ui_buy':'ui_ct_cvr'});
df_X_ui8 = df_X_ui8.join(df_X_ui1, how='outer').join(df_X_ui4, how='outer');
df_X_ui8.fillna(0, inplace=True);
df_X_ui8.ui_ct_cvr = df_X_ui8.ui_buy / df_X_ui8.ui_cart;
df_X_ui8.replace(np.inf, 0, inplace=True);
df_X_ui8.drop(['ui_buy', 'ui_cart'], axis=1, inplace=True);

df_X_ui9 = df_X_ui1.rename(columns={'ui_buy':'ui_buy_pr'});
df_X_ui9 = df_X_ui9.join(df_X_ui1, how='outer').join(df2.set_index('item_id'), how='left');
df_X_ui9 = df_X_ui9.reset_index(level=[0, 1]).set_index(['user_id', 'item_category']);
df_X_ui9 = df_X_ui9.join(df_X_uc1, how='outer');
df_X_ui9.ui_buy_pr = df_X_ui9.ui_buy / df_X_ui9.uc_buy;
df_X_ui9 = df_X_ui9.reset_index(level=[0, 1]).set_index(['user_id', 'item_id']);
df_X_ui9.drop(['ui_buy', 'item_category', 'uc_buy'], axis=1, inplace=True);

df_X_ui10 = df_X_ui2.rename(columns={'ui_click':'ui_ck_pr'});
df_X_ui10 = df_X_ui10.join(df_X_ui2, how='outer').join(df2.set_index('item_id'), how='left');
df_X_ui10 = df_X_ui10.reset_index(level=[0, 1]).set_index(['user_id', 'item_category']);
df_X_ui10 = df_X_ui10.join(df_X_uc2, how='outer');
df_X_ui10.ui_ck_pr = df_X_ui10.ui_click / df_X_ui10.uc_click;
df_X_ui10 = df_X_ui10.reset_index(level=[0, 1]).set_index(['user_id', 'item_id']);
df_X_ui10.drop(['ui_click', 'item_category', 'uc_click'], axis=1, inplace=True);

df_X_ui11 = df_X_ui3.rename(columns={'ui_collec':'ui_cl_pr'});
df_X_ui11 = df_X_ui11.join(df_X_ui3, how='outer').join(df2.set_index('item_id'), how='left');
df_X_ui11 = df_X_ui11.reset_index(level=[0, 1]).set_index(['user_id', 'item_category']);
df_X_ui11 = df_X_ui11.join(df_X_uc3, how='outer');
df_X_ui11.ui_cl_pr = df_X_ui11.ui_collec / df_X_ui11.uc_collec;
df_X_ui11 = df_X_ui11.reset_index(level=[0, 1]).set_index(['user_id', 'item_id']);
df_X_ui11.drop(['ui_collec', 'item_category', 'uc_collec'], axis=1, inplace=True);

df_X_ui12 = df_X_ui4.rename(columns={'ui_cart':'ui_ct_pr'});
df_X_ui12 = df_X_ui12.join(df_X_ui4, how='outer').join(df2.set_index('item_id'), how='left');
df_X_ui12 = df_X_ui12.reset_index(level=[0, 1]).set_index(['user_id', 'item_category']);
df_X_ui12 = df_X_ui12.join(df_X_uc4, how='outer');
df_X_ui12.ui_ct_pr = df_X_ui12.ui_cart / df_X_ui12.uc_cart;
df_X_ui12 = df_X_ui12.reset_index(level=[0, 1]).set_index(['user_id', 'item_id']);
df_X_ui12.drop(['ui_cart', 'item_category', 'uc_cart'], axis=1, inplace=True);

df_X_ui13 = df_clean.loc[df_clean.behavior_type==4].groupby(['user_id', 'item_id', 'time']).size().reset_index(name='ui_co_buy_pre');
df_X_ui13.ui_co_buy_pre = (df_X_ui13.ui_co_buy_pre!=0).astype(int);
df_X_ui13s = df_clean.loc[df_clean.behavior_type==4];
df_X_ui13s.time = df_X_ui13s.time.sub(np.timedelta64(1, 'D'));
df_X_ui13s = df_X_ui13s.groupby(['user_id', 'item_id', 'time']).size().reset_index(name='ui_buy_nextday');
df_X_ui13s.ui_buy_nextday = (df_X_ui13s.ui_buy_nextday!=0).astype(int);
df_X_ui13 = df_X_ui13.set_index(['user_id', 'item_id', 'time']).join(df_X_ui13s.set_index(['user_id', 'item_id', 'time']), how='outer');
df_X_ui13.fillna(0, inplace=True);
df_X_ui13 = df_X_ui13.astype(bool);
df_X_ui13.ui_co_buy_pre = (df_X_ui13.ui_co_buy_pre & df_X_ui13.ui_buy_nextday);
df_X_ui13.ui_co_buy_pre = df_X_ui13.ui_co_buy_pre.astype(int);
df_X_ui13.drop(['ui_buy_nextday'], axis=1, inplace=True);
df_X_ui13.drop([np.datetime64('2014-11-17'), np.datetime64('2014-12-18')], axis=0, level=1, inplace=True);
df_X_ui13 = df_X_ui13.reset_index(level=[0, 1, 2]);
df_X_ui13.drop(['time'], axis=1, inplace=True);
df_X_ui13 = df_X_ui13.groupby(['user_id', 'item_id']).mean();

##for user X time
#for today
df_X_ut1_today = df_clean.loc[df_clean.behavior_type==4].groupby(['user_id', 'time']).size().reset_index(name='ut_buy_today').set_index(['user_id', 'time']);
df_X_ut2_today = df_clean.loc[df_clean.behavior_type==1].groupby(['user_id', 'time']).size().reset_index(name='ut_click_today').set_index(['user_id', 'time']);
df_X_ut3_today = df_clean.loc[df_clean.behavior_type==2].groupby(['user_id', 'time']).size().reset_index(name='ut_collec_today').set_index(['user_id', 'time']);
df_X_ut4_today = df_clean.loc[df_clean.behavior_type==3].groupby(['user_id', 'time']).size().reset_index(name='ut_cart_today').set_index(['user_id', 'time']);

df_X_ut5_today = df_X_ut1_today.rename(columns={'ut_buy_today':'ut_bh_cvr_today'});
df_X_ut5_today = df_X_ut5_today.join(df_X_ut1_today, how='outer').join(df_X_ut2_today, how='outer').join(df_X_ut3_today, how='outer').join(df_X_ut4_today, how='outer');
df_X_ut5_today.fillna(0, inplace=True);
df_X_ut5_today.ut_bh_cvr_today = df_X_ut5_today.ut_buy_today / (df_X_ut5_today.ut_buy_today + df_X_ut5_today.ut_click_today + df_X_ut5_today.ut_collec_today + df_X_ut5_today.ut_cart_today);
df_X_ut5_today.drop(['ut_buy_today', 'ut_click_today', 'ut_collec_today', 'ut_cart_today'], axis=1, inplace=True);

df_X_ut6_today = df_X_ut1_today.rename(columns={'ut_buy_today':'ut_ck_cvr_today'});
df_X_ut6_today = df_X_ut6_today.join(df_X_ut1_today, how='outer').join(df_X_ut2_today, how='outer');
df_X_ut6_today.fillna(0, inplace=True);
df_X_ut6_today.ut_ck_cvr_today = df_X_ut6_today.ut_buy_today / df_X_ut6_today.ut_click_today;
df_X_ut6_today.replace(np.inf, 0, inplace=True);
df_X_ut6_today.drop(['ut_buy_today', 'ut_click_today'], axis=1, inplace=True);

df_X_ut7_today = df_X_ut1_today.rename(columns={'ut_buy_today':'ut_cl_cvr_today'});
df_X_ut7_today = df_X_ut7_today.join(df_X_ut1_today, how='outer').join(df_X_ut3_today, how='outer');
df_X_ut7_today.fillna(0, inplace=True);
df_X_ut7_today.ut_cl_cvr_today = df_X_ut7_today.ut_buy_today / df_X_ut7_today.ut_collec_today;
df_X_ut7_today.replace(np.inf, 0, inplace=True);
df_X_ut7_today.drop(['ut_buy_today', 'ut_collec_today'], axis=1, inplace=True);

df_X_ut8_today = df_X_ut1_today.rename(columns={'ut_buy_today':'ut_ct_cvr_today'});
df_X_ut8_today = df_X_ut8_today.join(df_X_ut1_today, how='outer').join(df_X_ut4_today, how='outer');
df_X_ut8_today.fillna(0, inplace=True);
df_X_ut8_today.ut_ct_cvr_today = df_X_ut8_today.ut_buy_today / df_X_ut8_today.ut_cart_today;
df_X_ut8_today.replace(np.inf, 0, inplace=True);
df_X_ut8_today.drop(['ut_buy_today', 'ut_cart_today'], axis=1, inplace=True);

df_X_ut_day = [];
for i in range(0, 11):
	df_X_ut_iday = df_clean.loc[:, ['user_id', 'item_id', 'time', 'behavior_type']];
	df_X_ut_iday.time = df_X_ut_iday.time.add(np.timedelta64(int(i), 'D'));
	df_X_utn = [];
	df_X_utn.append(df_X_ut_iday.loc[df_X_ut_iday.behavior_type==4].groupby(['user_id', 'time']).size().reset_index(name='ut_buy_' + str(i) + 'day'));
	df_X_utn.append(df_X_ut_iday.loc[df_X_ut_iday.behavior_type==1].groupby(['user_id', 'time']).size().reset_index(name='ut_click_' + str(i) + 'day'));
	df_X_utn.append(df_X_ut_iday.loc[df_X_ut_iday.behavior_type==2].groupby(['user_id', 'time']).size().reset_index(name='ut_collec_' + str(i) + 'day'));
	df_X_utn.append(df_X_ut_iday.loc[df_X_ut_iday.behavior_type==3].groupby(['user_id', 'time']).size().reset_index(name='ut_cart_' + str(i) + 'day'));
	df_X_ut_day.append(df_X_utn);
df_X_ut1s_010day = df_X_ut_day[0][0].rename(columns={'ut_buy_0day':'ut_buy_010day'}).set_index(['user_id', 'time']);
df_X_ut2s_010day = df_X_ut_day[0][1].rename(columns={'ut_click_0day':'ut_click_010day'}).set_index(['user_id', 'time']);
df_X_ut3s_010day = df_X_ut_day[0][2].rename(columns={'ut_collec_0day':'ut_collec_010day'}).set_index(['user_id', 'time']);
df_X_ut4s_010day = df_X_ut_day[0][3].rename(columns={'ut_cart_0day':'ut_cart_010day'}).set_index(['user_id', 'time']);
df_X_ut1s_010day.ut_buy_010day = 0;
df_X_ut2s_010day.ut_click_010day = 0;
df_X_ut3s_010day.ut_collec_010day = 0;
df_X_ut4s_010day.ut_cart_010day = 0;
for i in range(0, 11):
	df_X_ut1s_010day = df_X_ut1s_010day.join(df_X_ut_day[i][0].set_index(['user_id', 'time']), how='outer');
	df_X_ut2s_010day = df_X_ut2s_010day.join(df_X_ut_day[i][1].set_index(['user_id', 'time']), how='outer');
	df_X_ut3s_010day = df_X_ut3s_010day.join(df_X_ut_day[i][2].set_index(['user_id', 'time']), how='outer');
	df_X_ut4s_010day = df_X_ut4s_010day.join(df_X_ut_day[i][3].set_index(['user_id', 'time']), how='outer');
df_X_ut1s_010day.fillna(0, inplace=True);
df_X_ut2s_010day.fillna(0, inplace=True);
df_X_ut3s_010day.fillna(0, inplace=True);
df_X_ut4s_010day.fillna(0, inplace=True);
for i in range(1, 12):
	df_X_ut1s_010day.ix[:, 0] = df_X_ut1s_010day.ix[:, 0] + df_X_ut1s_010day.ix[:, i] * (0.65 ** (np.log((i-1)**2 + 1)));
	df_X_ut2s_010day.ix[:, 0] = df_X_ut2s_010day.ix[:, 0] + df_X_ut2s_010day.ix[:, i] * (0.65 ** (np.log((i-1)**2 + 1)));
	df_X_ut3s_010day.ix[:, 0] = df_X_ut3s_010day.ix[:, 0] + df_X_ut3s_010day.ix[:, i] * (0.65 ** (np.log((i-1)**2 + 1)));
	df_X_ut4s_010day.ix[:, 0] = df_X_ut4s_010day.ix[:, 0] + df_X_ut4s_010day.ix[:, i] * (0.65 ** (np.log((i-1)**2 + 1)));
df_X_ut1s_010day.drop(df_X_ut1s_010day.columns[[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]], axis=1, inplace=True);
df_X_ut2s_010day.drop(df_X_ut2s_010day.columns[[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]], axis=1, inplace=True);
df_X_ut3s_010day.drop(df_X_ut3s_010day.columns[[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]], axis=1, inplace=True);
df_X_ut4s_010day.drop(df_X_ut4s_010day.columns[[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]], axis=1, inplace=True);

df_X_ut9_today = df_X_ut1_today.rename(columns={'ut_buy_today':'ut_buy_pr_today'});
df_X_ut9_today = df_X_ut9_today.join(df_X_ut1_today, how='outer').join(df_X_ut1s_010day, how='left');
df_X_ut9_today.fillna(0, inplace=True);
df_X_ut9_today.ut_buy_pr_today = df_X_ut9_today.ut_buy_today / df_X_ut9_today.ut_buy_010day;
df_X_ut9_today.drop(df_X_ut9_today.columns[[1, 2]],axis=1, inplace=True);

df_X_ut10_today = df_X_ut2_today.rename(columns={'ut_click_today':'ut_ck_pr_today'});
df_X_ut10_today = df_X_ut10_today.join(df_X_ut2_today, how='outer').join(df_X_ut2s_010day, how='left');
df_X_ut10_today.fillna(0, inplace=True);
df_X_ut10_today.ut_ck_pr_today = df_X_ut10_today.ut_click_today / df_X_ut10_today.ut_click_010day;
df_X_ut10_today.drop(df_X_ut10_today.columns[[1, 2]],axis=1, inplace=True);

df_X_ut11_today = df_X_ut3_today.rename(columns={'ut_collec_today':'ut_cl_pr_today'});
df_X_ut11_today = df_X_ut11_today.join(df_X_ut3_today, how='outer').join(df_X_ut3s_010day, how='left');
df_X_ut11_today.fillna(0, inplace=True);
df_X_ut11_today.ut_cl_pr_today = df_X_ut11_today.ut_collec_today / df_X_ut11_today.ut_collec_010day;
df_X_ut11_today.drop(df_X_ut11_today.columns[[1, 2]],axis=1, inplace=True);

df_X_ut12_today = df_X_ut4_today.rename(columns={'ut_cart_today':'ut_ct_pr_today'});
df_X_ut12_today = df_X_ut12_today.join(df_X_ut4_today, how='outer').join(df_X_ut4s_010day, how='left');
df_X_ut12_today.fillna(0, inplace=True);
df_X_ut12_today.ut_ct_pr_today = df_X_ut12_today.ut_cart_today / df_X_ut12_today.ut_cart_010day;
df_X_ut12_today.drop(df_X_ut12_today.columns[[1, 2]],axis=1, inplace=True);

#for 1 day before
df_X_ut1_1day = df_X_ut1_today.reset_index(level=[0, 1]);
df_X_ut1_1day.time = df_X_ut1_1day.time.add(np.timedelta64(1, 'D'));
df_X_ut1_1day = df_X_ut1_1day.rename(columns={'ut_buy_today':'ut_buy_1day'});
df_X_ut1_1day = df_X_ut1_1day.set_index(['user_id', 'time']);

df_X_ut2_1day = df_X_ut2_today.reset_index(level=[0, 1]);
df_X_ut2_1day.time = df_X_ut2_1day.time.add(np.timedelta64(1, 'D'));
df_X_ut2_1day = df_X_ut2_1day.rename(columns={'ut_click_today':'ut_click_1day'});
df_X_ut2_1day = df_X_ut2_1day.set_index(['user_id', 'time']);

df_X_ut3_1day = df_X_ut3_today.reset_index(level=[0, 1]);
df_X_ut3_1day.time = df_X_ut3_1day.time.add(np.timedelta64(1, 'D'));
df_X_ut3_1day = df_X_ut3_1day.rename(columns={'ut_collec_today':'ut_collec_1day'});
df_X_ut3_1day = df_X_ut3_1day.set_index(['user_id', 'time']);

df_X_ut4_1day = df_X_ut4_today.reset_index(level=[0, 1]);
df_X_ut4_1day.time = df_X_ut4_1day.time.add(np.timedelta64(1, 'D'));
df_X_ut4_1day = df_X_ut4_1day.rename(columns={'ut_cart_today':'ut_cart_1day'});
df_X_ut4_1day = df_X_ut4_1day.set_index(['user_id', 'time']);

df_X_ut5_1day = df_X_ut5_today.reset_index(level=[0, 1]);
df_X_ut5_1day.time = df_X_ut5_1day.time.add(np.timedelta64(1, 'D'));
df_X_ut5_1day = df_X_ut5_1day.rename(columns={'ut_bh_cvr_today':'ut_bh_cvr_1day'});
df_X_ut5_1day = df_X_ut5_1day.set_index(['user_id', 'time']);

df_X_ut6_1day = df_X_ut6_today.reset_index(level=[0, 1]);
df_X_ut6_1day.time = df_X_ut6_1day.time.add(np.timedelta64(1, 'D'));
df_X_ut6_1day = df_X_ut6_1day.rename(columns={'ut_ck_cvr_today':'ut_ck_cvr_1day'});
df_X_ut6_1day = df_X_ut6_1day.set_index(['user_id', 'time']);

df_X_ut7_1day = df_X_ut7_today.reset_index(level=[0, 1]);
df_X_ut7_1day.time = df_X_ut7_1day.time.add(np.timedelta64(1, 'D'));
df_X_ut7_1day = df_X_ut7_1day.rename(columns={'ut_cl_cvr_today':'ut_cl_cvr_1day'});
df_X_ut7_1day = df_X_ut7_1day.set_index(['user_id', 'time']);

df_X_ut8_1day = df_X_ut8_today.reset_index(level=[0, 1]);
df_X_ut8_1day.time = df_X_ut8_1day.time.add(np.timedelta64(1, 'D'));
df_X_ut8_1day = df_X_ut8_1day.rename(columns={'ut_ct_cvr_today':'ut_ct_cvr_1day'});
df_X_ut8_1day = df_X_ut8_1day.set_index(['user_id', 'time']);

df_X_ut9_1day = df_X_ut9_today.reset_index(level=[0, 1]);
df_X_ut9_1day.time = df_X_ut9_1day.time.add(np.timedelta64(1, 'D'));
df_X_ut9_1day = df_X_ut9_1day.rename(columns={'ut_buy_pr_today':'ut_buy_pr_1day'});
df_X_ut9_1day = df_X_ut9_1day.set_index(['user_id', 'time']);

df_X_ut10_1day = df_X_ut10_today.reset_index(level=[0, 1]);
df_X_ut10_1day.time = df_X_ut10_1day.time.add(np.timedelta64(1, 'D'));
df_X_ut10_1day = df_X_ut10_1day.rename(columns={'ut_ck_pr_today':'ut_ck_pr_1day'});
df_X_ut10_1day = df_X_ut10_1day.set_index(['user_id', 'time']);

df_X_ut11_1day = df_X_ut11_today.reset_index(level=[0, 1]);
df_X_ut11_1day.time = df_X_ut11_1day.time.add(np.timedelta64(1, 'D'));
df_X_ut11_1day = df_X_ut11_1day.rename(columns={'ut_cl_pr_today':'ut_cl_pr_1day'});
df_X_ut11_1day = df_X_ut11_1day.set_index(['user_id', 'time']);

df_X_ut12_1day = df_X_ut12_today.reset_index(level=[0, 1]);
df_X_ut12_1day.time = df_X_ut12_1day.time.add(np.timedelta64(1, 'D'));
df_X_ut12_1day = df_X_ut12_1day.rename(columns={'ut_ct_pr_today':'ut_ct_pr_1day'});
df_X_ut12_1day = df_X_ut12_1day.set_index(['user_id', 'time']);

#for 2 days before
df_X_ut1_2day = df_X_ut1_today.reset_index(level=[0, 1]);
df_X_ut1_2day.time = df_X_ut1_2day.time.add(np.timedelta64(2, 'D'));
df_X_ut1_2day = df_X_ut1_2day.rename(columns={'ut_buy_today':'ut_buy_2day'});
df_X_ut1_2day = df_X_ut1_2day.set_index(['user_id', 'time']);

df_X_ut2_2day = df_X_ut2_today.reset_index(level=[0, 1]);
df_X_ut2_2day.time = df_X_ut2_2day.time.add(np.timedelta64(2, 'D'));
df_X_ut2_2day = df_X_ut2_2day.rename(columns={'ut_click_today':'ut_click_2day'});
df_X_ut2_2day = df_X_ut2_2day.set_index(['user_id', 'time']);

df_X_ut3_2day = df_X_ut3_today.reset_index(level=[0, 1]);
df_X_ut3_2day.time = df_X_ut3_2day.time.add(np.timedelta64(2, 'D'));
df_X_ut3_2day = df_X_ut3_2day.rename(columns={'ut_collec_today':'ut_collec_2day'});
df_X_ut3_2day = df_X_ut3_2day.set_index(['user_id', 'time']);

df_X_ut4_2day = df_X_ut4_today.reset_index(level=[0, 1]);
df_X_ut4_2day.time = df_X_ut4_2day.time.add(np.timedelta64(2, 'D'));
df_X_ut4_2day = df_X_ut4_2day.rename(columns={'ut_cart_today':'ut_cart_2day'});
df_X_ut4_2day = df_X_ut4_2day.set_index(['user_id', 'time']);

df_X_ut5_2day = df_X_ut5_today.reset_index(level=[0, 1]);
df_X_ut5_2day.time = df_X_ut5_2day.time.add(np.timedelta64(2, 'D'));
df_X_ut5_2day = df_X_ut5_2day.rename(columns={'ut_bh_cvr_today':'ut_bh_cvr_2day'});
df_X_ut5_2day = df_X_ut5_2day.set_index(['user_id', 'time']);

df_X_ut6_2day = df_X_ut6_today.reset_index(level=[0, 1]);
df_X_ut6_2day.time = df_X_ut6_2day.time.add(np.timedelta64(2, 'D'));
df_X_ut6_2day = df_X_ut6_2day.rename(columns={'ut_ck_cvr_today':'ut_ck_cvr_2day'});
df_X_ut6_2day = df_X_ut6_2day.set_index(['user_id', 'time']);

df_X_ut7_2day = df_X_ut7_today.reset_index(level=[0, 1]);
df_X_ut7_2day.time = df_X_ut7_2day.time.add(np.timedelta64(2, 'D'));
df_X_ut7_2day = df_X_ut7_2day.rename(columns={'ut_cl_cvr_today':'ut_cl_cvr_2day'});
df_X_ut7_2day = df_X_ut7_2day.set_index(['user_id', 'time']);

df_X_ut8_2day = df_X_ut8_today.reset_index(level=[0, 1]);
df_X_ut8_2day.time = df_X_ut8_2day.time.add(np.timedelta64(2, 'D'));
df_X_ut8_2day = df_X_ut8_2day.rename(columns={'ut_ct_cvr_today':'ut_ct_cvr_2day'});
df_X_ut8_2day = df_X_ut8_2day.set_index(['user_id', 'time']);

df_X_ut9_2day = df_X_ut9_today.reset_index(level=[0, 1]);
df_X_ut9_2day.time = df_X_ut9_2day.time.add(np.timedelta64(2, 'D'));
df_X_ut9_2day = df_X_ut9_2day.rename(columns={'ut_buy_pr_today':'ut_buy_pr_2day'});
df_X_ut9_2day = df_X_ut9_2day.set_index(['user_id', 'time']);

df_X_ut10_2day = df_X_ut10_today.reset_index(level=[0, 1]);
df_X_ut10_2day.time = df_X_ut10_2day.time.add(np.timedelta64(2, 'D'));
df_X_ut10_2day = df_X_ut10_2day.rename(columns={'ut_ck_pr_today':'ut_ck_pr_2day'});
df_X_ut10_2day = df_X_ut10_2day.set_index(['user_id', 'time']);

df_X_ut11_2day = df_X_ut11_today.reset_index(level=[0, 1]);
df_X_ut11_2day.time = df_X_ut11_2day.time.add(np.timedelta64(2, 'D'));
df_X_ut11_2day = df_X_ut11_2day.rename(columns={'ut_cl_pr_today':'ut_cl_pr_2day'});
df_X_ut11_2day = df_X_ut11_2day.set_index(['user_id', 'time']);

df_X_ut12_2day = df_X_ut12_today.reset_index(level=[0, 1]);
df_X_ut12_2day.time = df_X_ut12_2day.time.add(np.timedelta64(2, 'D'));
df_X_ut12_2day = df_X_ut12_2day.rename(columns={'ut_ct_pr_today':'ut_ct_pr_2day'});
df_X_ut12_2day = df_X_ut12_2day.set_index(['user_id', 'time']);

#for 3-10 days before
df_X_ut_day = [];
for i in range(3, 11):
	df_X_ut_iday = df_clean.loc[:, ['user_id', 'item_id', 'time', 'behavior_type']];
	df_X_ut_iday.time = df_X_ut_iday.time.add(np.timedelta64(int(i), 'D'));
	df_X_utn = [];
	df_X_utn.append(df_X_ut_iday.loc[df_X_ut_iday.behavior_type==4].groupby(['user_id', 'time']).size().reset_index(name='ut_buy_' + str(i) + 'day'));
	df_X_utn.append(df_X_ut_iday.loc[df_X_ut_iday.behavior_type==1].groupby(['user_id', 'time']).size().reset_index(name='ut_click_' + str(i) + 'day'));
	df_X_utn.append(df_X_ut_iday.loc[df_X_ut_iday.behavior_type==2].groupby(['user_id', 'time']).size().reset_index(name='ut_collec_' + str(i) + 'day'));
	df_X_utn.append(df_X_ut_iday.loc[df_X_ut_iday.behavior_type==3].groupby(['user_id', 'time']).size().reset_index(name='ut_cart_' + str(i) + 'day'));
	df_X_ut_day.append(df_X_utn);
df_X_ut1_310day = df_X_ut_day[0][0].rename(columns={'ut_buy_3day':'ut_buy_310day'}).set_index(['user_id', 'time']);
df_X_ut2_310day = df_X_ut_day[0][1].rename(columns={'ut_click_3day':'ut_click_310day'}).set_index(['user_id', 'time']);
df_X_ut3_310day = df_X_ut_day[0][2].rename(columns={'ut_collec_3day':'ut_collec_310day'}).set_index(['user_id', 'time']);
df_X_ut4_310day = df_X_ut_day[0][3].rename(columns={'ut_cart_3day':'ut_cart_310day'}).set_index(['user_id', 'time']);
df_X_ut1_310day.ut_buy_310day = 0;
df_X_ut2_310day.ut_click_310day = 0;
df_X_ut3_310day.ut_collec_310day = 0;
df_X_ut4_310day.ut_cart_310day = 0;
for i in range(0, 8):
    df_X_ut1_310day = df_X_ut1_310day.join(df_X_ut_day[i][0].set_index(['user_id', 'time']), how='outer');
    df_X_ut2_310day = df_X_ut2_310day.join(df_X_ut_day[i][1].set_index(['user_id', 'time']), how='outer');
    df_X_ut3_310day = df_X_ut3_310day.join(df_X_ut_day[i][2].set_index(['user_id', 'time']), how='outer');
    df_X_ut4_310day = df_X_ut4_310day.join(df_X_ut_day[i][3].set_index(['user_id', 'time']), how='outer');
df_X_ut1_310day.fillna(0, inplace=True);
df_X_ut2_310day.fillna(0, inplace=True);
df_X_ut3_310day.fillna(0, inplace=True);
df_X_ut4_310day.fillna(0, inplace=True);
for i in range(1, 9):
    df_X_ut1_310day.ix[:, 0] = df_X_ut1_310day.ix[:, 0] + df_X_ut1_310day.ix[:, i] * (0.65 ** (np.log((i+2)**2 + 1))); #time decay
    df_X_ut2_310day.ix[:, 0] = df_X_ut2_310day.ix[:, 0] + df_X_ut2_310day.ix[:, i] * (0.65 ** (np.log((i+2)**2 + 1))); 
    df_X_ut3_310day.ix[:, 0] = df_X_ut3_310day.ix[:, 0] + df_X_ut3_310day.ix[:, i] * (0.65 ** (np.log((i+2)**2 + 1))); 
    df_X_ut4_310day.ix[:, 0] = df_X_ut4_310day.ix[:, 0] + df_X_ut4_310day.ix[:, i] * (0.65 ** (np.log((i+2)**2 + 1))); 
df_X_ut1_310day.drop(df_X_ut1_310day.columns[[1, 2, 3, 4, 5, 6, 7, 8]], axis=1, inplace=True);
df_X_ut2_310day.drop(df_X_ut2_310day.columns[[1, 2, 3, 4, 5, 6, 7, 8]], axis=1, inplace=True);
df_X_ut3_310day.drop(df_X_ut3_310day.columns[[1, 2, 3, 4, 5, 6, 7, 8]], axis=1, inplace=True);
df_X_ut4_310day.drop(df_X_ut4_310day.columns[[1, 2, 3, 4, 5, 6, 7, 8]], axis=1, inplace=True);

df_X_ut5_310day = df_X_ut1_310day.rename(columns={'ut_buy_310day':'ut_bh_cvr_310day'});
df_X_ut5_310day = df_X_ut5_310day.join(df_X_ut1_310day, how='outer').join(df_X_ut2_310day, how='outer').join(df_X_ut3_310day, how='outer').join(df_X_ut4_310day, how='outer');
df_X_ut5_310day.fillna(0, inplace=True);
df_X_ut5_310day.ut_bh_cvr_310day = df_X_ut5_310day.ut_buy_310day / (df_X_ut5_310day.ut_buy_310day + df_X_ut5_310day.ut_click_310day + df_X_ut5_310day.ut_collec_310day + df_X_ut5_310day.ut_cart_310day);
df_X_ut5_310day.drop(df_X_ut5_310day.columns[[1, 2, 3, 4]], axis=1, inplace=True);

df_X_ut6_310day = df_X_ut1_310day.rename(columns={'ut_buy_310day':'ut_ck_cvr_310day'});
df_X_ut6_310day = df_X_ut6_310day.join(df_X_ut1_310day, how='outer').join(df_X_ut2_310day, how='outer');
df_X_ut6_310day.fillna(0, inplace=True);
df_X_ut6_310day.ut_ck_cvr_310day = df_X_ut6_310day.ut_buy_310day / df_X_ut6_310day.ut_click_310day;
df_X_ut6_310day.replace(np.inf, 0, inplace=True);
df_X_ut6_310day.drop(df_X_ut6_310day.columns[[1, 2]], axis=1, inplace=True);

df_X_ut7_310day = df_X_ut1_310day.rename(columns={'ut_buy_310day':'ut_cl_cvr_310day'});
df_X_ut7_310day = df_X_ut7_310day.join(df_X_ut1_310day, how='outer').join(df_X_ut3_310day, how='outer');
df_X_ut7_310day.fillna(0, inplace=True);
df_X_ut7_310day.ut_cl_cvr_310day = df_X_ut7_310day.ut_buy_310day / df_X_ut7_310day.ut_collec_310day;
df_X_ut7_310day.replace(np.inf, 0, inplace=True);
df_X_ut7_310day.drop(df_X_ut7_310day.columns[[1, 2]], axis=1, inplace=True);

df_X_ut8_310day = df_X_ut1_310day.rename(columns={'ut_buy_310day':'ut_ct_cvr_310day'});
df_X_ut8_310day = df_X_ut8_310day.join(df_X_ut1_310day, how='outer').join(df_X_ut4_310day, how='outer');
df_X_ut8_310day.fillna(0, inplace=True);
df_X_ut8_310day.ut_ct_cvr_310day = df_X_ut8_310day.ut_buy_310day / df_X_ut8_310day.ut_cart_310day;
df_X_ut8_310day.replace(np.inf, 0, inplace=True);
df_X_ut8_310day.drop(df_X_ut8_310day.columns[[1, 2]], axis=1, inplace=True);

df_X_ut9_310day = df_X_ut1_310day.rename(columns={'ut_buy_310day':'ut_buy_pr_310day'});
df_X_ut9_310day = df_X_ut9_310day.join(df_X_ut1_310day, how='outer').join(df_X_u1, how='left');
df_X_ut9_310day.ut_buy_pr_310day = df_X_ut9_310day.ut_buy_310day / df_X_ut9_310day.ur_buy;
df_X_ut9_310day.drop(df_X_ut9_310day.columns[[1, 2]],axis=1, inplace=True);

df_X_ut10_310day = df_X_ut2_310day.rename(columns={'ut_click_310day':'ut_ck_pr_310day'});
df_X_ut10_310day = df_X_ut10_310day.join(df_X_ut2_310day, how='outer').join(df_X_u2, how='left');
df_X_ut10_310day.ut_ck_pr_310day = df_X_ut10_310day.ut_click_310day / df_X_ut10_310day.ur_click;
df_X_ut10_310day.drop(df_X_ut10_310day.columns[[1, 2]],axis=1, inplace=True);

df_X_ut11_310day = df_X_ut3_310day.rename(columns={'ut_collec_310day':'ut_cl_pr_310day'});
df_X_ut11_310day = df_X_ut11_310day.join(df_X_ut3_310day, how='outer').join(df_X_u3, how='left');
df_X_ut11_310day.ut_cl_pr_310day = df_X_ut11_310day.ut_collec_310day / df_X_ut11_310day.ur_collec;
df_X_ut11_310day.drop(df_X_ut11_310day.columns[[1, 2]],axis=1, inplace=True);

df_X_ut12_310day = df_X_ut4_310day.rename(columns={'ut_cart_310day':'ut_ct_pr_310day'});
df_X_ut12_310day = df_X_ut12_310day.join(df_X_ut4_310day, how='outer').join(df_X_u4, how='left');
df_X_ut12_310day.ut_ct_pr_310day = df_X_ut12_310day.ut_cart_310day / df_X_ut12_310day.ur_cart;
df_X_ut12_310day.drop(df_X_ut12_310day.columns[[1, 2]],axis=1, inplace=True);

#for 0-10 days before
df_X_ut_day = [];
for i in range(0, 11):
	df_X_ut_iday = df_clean.loc[:, ['user_id', 'item_id', 'time', 'behavior_type']];
	df_X_ut_iday.time = df_X_ut_iday.time.add(np.timedelta64(int(i), 'D'));
	df_X_ut_day.append(df_X_ut_iday.loc[df_X_ut_iday.behavior_type==4].groupby(['user_id', 'item_id', 'time']).size().reset_index(name='ut_buy_' + str(i) + 'day'));
df_X_ut1_010day = df_X_ut_day[0].rename(columns={'ut_buy_0day':'ut_buy_dp_010day'}).set_index(['user_id', 'item_id', 'time']);
df_X_ut1_010day.ut_buy_dp_010day = 0;
for i in range(0, 11):
	df_X_ut1_010day = df_X_ut1_010day.join(df_X_ut_day[i].set_index(['user_id', 'item_id', 'time']), how='outer');
df_X_ut1_010day.fillna(0, inplace=True);
df_X_ut1_010day = (df_X_ut1_010day!=0).astype(int);
for i in range(1, 12):
	df_X_ut1_010day.ix[:, 0] = df_X_ut1_010day.ix[:, 0] + df_X_ut1_010day.ix[:, i];
df_X_ut1_010day.drop(df_X_ut1_010day.columns[[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]], axis=1, inplace=True);
df_X_ut1_010day.ut_buy_dp_010day = (df_X_ut1_010day.ut_buy_dp_010day!=1);
df_X_ut1_010day.ut_buy_dp_010day = df_X_ut1_010day.ut_buy_dp_010day.astype(int);
df_X_ut1_010day = df_X_ut1_010day.reset_index(level=[0, 1, 2]);
df_X_ut1_010day.drop(df_X_ut1_010day.columns[[1]], axis=1, inplace=True);
df_X_ut1_010day = df_X_ut1_010day.groupby(['user_id', 'time']).mean();

df_X_ut_day = [];
for i in range(0, 11):
	df_X_ut_iday = df_clean.loc[:, ['user_id', 'item_id', 'time', 'behavior_type']];
	df_X_ut_iday.time = df_X_ut_iday.time.add(np.timedelta64(int(i), 'D'));
	df_X_ut_day.append(df_X_ut_iday.loc[df_X_ut_iday.behavior_type==4].groupby(['user_id', 'time']).size().reset_index(name='ut_buy_' + str(i) + 'day'));
df_X_ut2_010day = df_X_ut_day[0].rename(columns={'ut_buy_0day':'ut_co_buy_pre_010day'}).set_index(['user_id', 'time']);
df_X_ut2_010day.ut_co_buy_pre_010day = 0;
for i in range(0, 11):
	df_X_ut2_010day = df_X_ut2_010day.join(df_X_ut_day[i].set_index(['user_id', 'time']), how='outer');
df_X_ut2_010day.fillna(0, inplace=True);
df_X_ut2_010day = (df_X_ut2_010day!=0).astype(int);
for i in range(1, 12):
	df_X_ut2_010day.ix[:, 0] = df_X_ut2_010day.ix[:, 0] + df_X_ut2_010day.ix[:, i];
for i in range(1, 11):
	df_X_ut2_010day.ix[:, i] = (df_X_ut2_010day.ix[:, i].astype(bool) & df_X_ut2_010day.ix[:, i+1].astype(bool));
	df_X_ut2_010day.ix[:, i] = df_X_ut2_010day.ix[:, i].astype(int);
for i in range(2, 11):
	df_X_ut2_010day.ix[:, 1] = df_X_ut2_010day.ix[:, 1] + df_X_ut2_010day.ix[:, i];
df_X_ut2_010day.ix[:, 0] = df_X_ut2_010day.ix[:, 1] / df_X_ut2_010day.ix[:, 0];
df_X_ut2_010day.drop(df_X_ut2_010day.columns[[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]], axis=1, inplace=True);

#for the while time
df_X_ut1 = df_clean.loc[:, ['user_id', 'item_id', 'time', 'behavior_type']].groupby(['user_id', 'time']).size().reset_index(name='ut_bh');
df_X_ut1.drop(['ut_bh'], axis=1, inplace=True);
df_X_ut1m = df_clean.loc[df_clean.behavior_type==4].groupby(['user_id', 'time']).size().reset_index(name='ut_buy');
df_X_ut1m.drop(['ut_buy'], axis=1, inplace=True);
df_X_ut1 = df_X_ut1.join(df_X_ut1m.set_index(['user_id']), on='user_id', rsuffix='_m', how='left');
df_X_ut1 = df_X_ut1.set_index(['user_id', 'time'], drop=False);
df_X_ut1.time_m = df_X_ut1.time - df_X_ut1.time_m;
df_X_ut1 = df_X_ut1.loc[df_X_ut1.time_m>np.timedelta64(0, 'D')];
df_X_ut1 = df_X_ut1.groupby(['user_id', 'time']).min();
df_X_ut1 = df_X_ut1.rename(columns={'time_m':'ut_buy_lasttime'});
df_X_ut1.ut_buy_lasttime = df_X_ut1.ut_buy_lasttime.dt.days;

df_X_ut2 = df_clean.loc[:, ['user_id', 'item_id', 'time', 'behavior_type']].groupby(['user_id', 'time']).size().reset_index(name='ut_bh');
df_X_ut2.drop(['ut_bh'], axis=1, inplace=True);
df_X_ut2m = df_X_ut2.loc[:, ['user_id', 'time']];
df_X_ut2m.index = df_X_ut2m.index + 1;
df_X_ut2 = df_X_ut2.join(df_X_ut2m, rsuffix='_m', how='left');
df_X_ut2 = df_X_ut2.set_index(['user_id', 'time'], drop=False);
df_X_ut2.user_id = df_X_ut2.user_id - df_X_ut2.user_id_m;
df_X_ut2.time = df_X_ut2.time - df_X_ut2.time_m;
df_X_ut2 = df_X_ut2.loc[df_X_ut2.user_id==0];
df_X_ut2 = df_X_ut2.rename(columns={'time':'ut_bh_lasttime'});
df_X_ut2.ut_bh_lasttime = df_X_ut2.ut_bh_lasttime.dt.days;
df_X_ut2.drop(['user_id', 'user_id_m', 'time_m'], axis=1, inplace=True);

##for item(category) X time
#for today
df_X_ct1_today = df_clean.loc[df_clean.behavior_type==4].groupby(['item_category', 'time']).size().reset_index(name='ct_buy_today').set_index(['item_category', 'time']);
df_X_ct2_today = df_clean.loc[df_clean.behavior_type==1].groupby(['item_category', 'time']).size().reset_index(name='ct_click_today').set_index(['item_category', 'time']);
df_X_ct3_today = df_clean.loc[df_clean.behavior_type==2].groupby(['item_category', 'time']).size().reset_index(name='ct_collec_today').set_index(['item_category', 'time']);
df_X_ct4_today = df_clean.loc[df_clean.behavior_type==3].groupby(['item_category', 'time']).size().reset_index(name='ct_cart_today').set_index(['item_category', 'time']);

df_X_ct5_today = df_X_ct1_today.rename(columns={'ct_buy_today':'ct_bh_cvr_today'});
df_X_ct5_today = df_X_ct5_today.join(df_X_ct1_today, how='outer').join(df_X_ct2_today, how='outer').join(df_X_ct3_today, how='outer').join(df_X_ct4_today, how='outer');
df_X_ct5_today.fillna(0, inplace=True);
df_X_ct5_today.ct_bh_cvr_today = df_X_ct5_today.ct_buy_today / (df_X_ct5_today.ct_buy_today + df_X_ct5_today.ct_click_today + df_X_ct5_today.ct_collec_today + df_X_ct5_today.ct_cart_today);
df_X_ct5_today.drop(['ct_buy_today', 'ct_click_today', 'ct_collec_today', 'ct_cart_today'], axis=1, inplace=True);

df_X_ct6_today = df_X_ct1_today.rename(columns={'ct_buy_today':'ct_ck_cvr_today'});
df_X_ct6_today = df_X_ct6_today.join(df_X_ct1_today, how='outer').join(df_X_ct2_today, how='outer');
df_X_ct6_today.fillna(0, inplace=True);
df_X_ct6_today.ct_ck_cvr_today = df_X_ct6_today.ct_buy_today / df_X_ct6_today.ct_click_today;
df_X_ct6_today.replace(np.inf, 0, inplace=True);
df_X_ct6_today.drop(['ct_buy_today', 'ct_click_today'], axis=1, inplace=True);

df_X_ct7_today = df_X_ct1_today.rename(columns={'ct_buy_today':'ct_cl_cvr_today'});
df_X_ct7_today = df_X_ct7_today.join(df_X_ct1_today, how='outer').join(df_X_ct3_today, how='outer');
df_X_ct7_today.fillna(0, inplace=True);
df_X_ct7_today.ct_cl_cvr_today = df_X_ct7_today.ct_buy_today / df_X_ct7_today.ct_collec_today;
df_X_ct7_today.replace(np.inf, 0, inplace=True);
df_X_ct7_today.drop(['ct_buy_today', 'ct_collec_today'], axis=1, inplace=True);

df_X_ct8_today = df_X_ct1_today.rename(columns={'ct_buy_today':'ct_ct_cvr_today'});
df_X_ct8_today = df_X_ct8_today.join(df_X_ct1_today, how='outer').join(df_X_ct4_today, how='outer');
df_X_ct8_today.fillna(0, inplace=True);
df_X_ct8_today.ct_ct_cvr_today = df_X_ct8_today.ct_buy_today / df_X_ct8_today.ct_cart_today;
df_X_ct8_today.replace(np.inf, 0, inplace=True);
df_X_ct8_today.drop(['ct_buy_today', 'ct_cart_today'], axis=1, inplace=True);

df_X_ct_day = [];
for i in range(0, 11):
	df_X_ct_iday = df_clean.loc[:, ['item_id', 'item_category', 'time', 'behavior_type']];
	df_X_ct_iday.time = df_X_ct_iday.time.add(np.timedelta64(int(i), 'D'));
	df_X_ctn = [];
	df_X_ctn.append(df_X_ct_iday.loc[df_X_ct_iday.behavior_type==4].groupby(['item_category', 'time']).size().reset_index(name='ct_buy_' + str(i) + 'day'));
	df_X_ctn.append(df_X_ct_iday.loc[df_X_ct_iday.behavior_type==1].groupby(['item_category', 'time']).size().reset_index(name='ct_click_' + str(i) + 'day'));
	df_X_ctn.append(df_X_ct_iday.loc[df_X_ct_iday.behavior_type==2].groupby(['item_category', 'time']).size().reset_index(name='ct_collec_' + str(i) + 'day'));
	df_X_ctn.append(df_X_ct_iday.loc[df_X_ct_iday.behavior_type==3].groupby(['item_category', 'time']).size().reset_index(name='ct_cart_' + str(i) + 'day'));
	df_X_ct_day.append(df_X_ctn);
df_X_ct1s_010day = df_X_ct_day[0][0].rename(columns={'ct_buy_0day':'ct_buy_010day'}).set_index(['item_category', 'time']);
df_X_ct2s_010day = df_X_ct_day[0][1].rename(columns={'ct_click_0day':'ct_click_010day'}).set_index(['item_category', 'time']);
df_X_ct3s_010day = df_X_ct_day[0][2].rename(columns={'ct_collec_0day':'ct_collec_010day'}).set_index(['item_category', 'time']);
df_X_ct4s_010day = df_X_ct_day[0][3].rename(columns={'ct_cart_0day':'ct_cart_010day'}).set_index(['item_category', 'time']);
df_X_ct1s_010day.ct_buy_010day = 0;
df_X_ct2s_010day.ct_click_010day = 0;
df_X_ct3s_010day.ct_collec_010day = 0;
df_X_ct4s_010day.ct_cart_010day = 0;
for i in range(0, 11):
	df_X_ct1s_010day = df_X_ct1s_010day.join(df_X_ct_day[i][0].set_index(['item_category', 'time']), how='outer');
	df_X_ct2s_010day = df_X_ct2s_010day.join(df_X_ct_day[i][1].set_index(['item_category', 'time']), how='outer');
	df_X_ct3s_010day = df_X_ct3s_010day.join(df_X_ct_day[i][2].set_index(['item_category', 'time']), how='outer');
	df_X_ct4s_010day = df_X_ct4s_010day.join(df_X_ct_day[i][3].set_index(['item_category', 'time']), how='outer');
df_X_ct1s_010day.fillna(0, inplace=True);
df_X_ct2s_010day.fillna(0, inplace=True);
df_X_ct3s_010day.fillna(0, inplace=True);
df_X_ct4s_010day.fillna(0, inplace=True);
for i in range(1, 12):
	df_X_ct1s_010day.ix[:, 0] = df_X_ct1s_010day.ix[:, 0] + df_X_ct1s_010day.ix[:, i] * (0.65 ** (np.log((i-1)**2 + 1)));
	df_X_ct2s_010day.ix[:, 0] = df_X_ct2s_010day.ix[:, 0] + df_X_ct2s_010day.ix[:, i] * (0.65 ** (np.log((i-1)**2 + 1)));
	df_X_ct3s_010day.ix[:, 0] = df_X_ct3s_010day.ix[:, 0] + df_X_ct3s_010day.ix[:, i] * (0.65 ** (np.log((i-1)**2 + 1)));
	df_X_ct4s_010day.ix[:, 0] = df_X_ct4s_010day.ix[:, 0] + df_X_ct4s_010day.ix[:, i] * (0.65 ** (np.log((i-1)**2 + 1)));
df_X_ct1s_010day.drop(df_X_ct1s_010day.columns[[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]], axis=1, inplace=True);
df_X_ct2s_010day.drop(df_X_ct2s_010day.columns[[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]], axis=1, inplace=True);
df_X_ct3s_010day.drop(df_X_ct3s_010day.columns[[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]], axis=1, inplace=True);
df_X_ct4s_010day.drop(df_X_ct4s_010day.columns[[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]], axis=1, inplace=True);

df_X_ct9_today = df_X_ct1_today.rename(columns={'ct_buy_today':'ct_buy_pr_today'});
df_X_ct9_today = df_X_ct9_today.join(df_X_ct1_today, how='outer').join(df_X_ct1s_010day, how='left');
df_X_ct9_today.fillna(0, inplace=True);
df_X_ct9_today.ct_buy_pr_today = df_X_ct9_today.ct_buy_today / df_X_ct9_today.ct_buy_010day;
df_X_ct9_today.drop(df_X_ct9_today.columns[[1, 2]],axis=1, inplace=True);

df_X_ct10_today = df_X_ct2_today.rename(columns={'ct_click_today':'ct_ck_pr_today'});
df_X_ct10_today = df_X_ct10_today.join(df_X_ct2_today, how='outer').join(df_X_ct2s_010day, how='left');
df_X_ct10_today.fillna(0, inplace=True);
df_X_ct10_today.ct_ck_pr_today = df_X_ct10_today.ct_click_today / df_X_ct10_today.ct_click_010day;
df_X_ct10_today.drop(df_X_ct10_today.columns[[1, 2]],axis=1, inplace=True);

df_X_ct11_today = df_X_ct3_today.rename(columns={'ct_collec_today':'ct_cl_pr_today'});
df_X_ct11_today = df_X_ct11_today.join(df_X_ct3_today, how='outer').join(df_X_ct3s_010day, how='left');
df_X_ct11_today.fillna(0, inplace=True);
df_X_ct11_today.ct_cl_pr_today = df_X_ct11_today.ct_collec_today / df_X_ct11_today.ct_collec_010day;
df_X_ct11_today.drop(df_X_ct11_today.columns[[1, 2]],axis=1, inplace=True);

df_X_ct12_today = df_X_ct4_today.rename(columns={'ct_cart_today':'ct_ct_pr_today'});
df_X_ct12_today = df_X_ct12_today.join(df_X_ct4_today, how='outer').join(df_X_ct4s_010day, how='left');
df_X_ct12_today.fillna(0, inplace=True);
df_X_ct12_today.ct_ct_pr_today = df_X_ct12_today.ct_cart_today / df_X_ct12_today.ct_cart_010day;
df_X_ct12_today.drop(df_X_ct12_today.columns[[1, 2]],axis=1, inplace=True);

#for 1 day before
df_X_ct1_1day = df_X_ct1_today.reset_index(level=[0, 1]);
df_X_ct1_1day.time = df_X_ct1_1day.time.add(np.timedelta64(1, 'D'));
df_X_ct1_1day = df_X_ct1_1day.rename(columns={'ct_buy_today':'ct_buy_1day'});
df_X_ct1_1day = df_X_ct1_1day.set_index(['item_category', 'time']);

df_X_ct2_1day = df_X_ct2_today.reset_index(level=[0, 1]);
df_X_ct2_1day.time = df_X_ct2_1day.time.add(np.timedelta64(1, 'D'));
df_X_ct2_1day = df_X_ct2_1day.rename(columns={'ct_click_today':'ct_click_1day'});
df_X_ct2_1day = df_X_ct2_1day.set_index(['item_category', 'time']);

df_X_ct3_1day = df_X_ct3_today.reset_index(level=[0, 1]);
df_X_ct3_1day.time = df_X_ct3_1day.time.add(np.timedelta64(1, 'D'));
df_X_ct3_1day = df_X_ct3_1day.rename(columns={'ct_collec_today':'ct_collec_1day'});
df_X_ct3_1day = df_X_ct3_1day.set_index(['item_category', 'time']);

df_X_ct4_1day = df_X_ct4_today.reset_index(level=[0, 1]);
df_X_ct4_1day.time = df_X_ct4_1day.time.add(np.timedelta64(1, 'D'));
df_X_ct4_1day = df_X_ct4_1day.rename(columns={'ct_cart_today':'ct_cart_1day'});
df_X_ct4_1day = df_X_ct4_1day.set_index(['item_category', 'time']);

df_X_ct5_1day = df_X_ct5_today.reset_index(level=[0, 1]);
df_X_ct5_1day.time = df_X_ct5_1day.time.add(np.timedelta64(1, 'D'));
df_X_ct5_1day = df_X_ct5_1day.rename(columns={'ct_bh_cvr_today':'ct_bh_cvr_1day'});
df_X_ct5_1day = df_X_ct5_1day.set_index(['item_category', 'time']);

df_X_ct6_1day = df_X_ct6_today.reset_index(level=[0, 1]);
df_X_ct6_1day.time = df_X_ct6_1day.time.add(np.timedelta64(1, 'D'));
df_X_ct6_1day = df_X_ct6_1day.rename(columns={'ct_ck_cvr_today':'ct_ck_cvr_1day'});
df_X_ct6_1day = df_X_ct6_1day.set_index(['item_category', 'time']);

df_X_ct7_1day = df_X_ct7_today.reset_index(level=[0, 1]);
df_X_ct7_1day.time = df_X_ct7_1day.time.add(np.timedelta64(1, 'D'));
df_X_ct7_1day = df_X_ct7_1day.rename(columns={'ct_cl_cvr_today':'ct_cl_cvr_1day'});
df_X_ct7_1day = df_X_ct7_1day.set_index(['item_category', 'time']);

df_X_ct8_1day = df_X_ct8_today.reset_index(level=[0, 1]);
df_X_ct8_1day.time = df_X_ct8_1day.time.add(np.timedelta64(1, 'D'));
df_X_ct8_1day = df_X_ct8_1day.rename(columns={'ct_ct_cvr_today':'ct_ct_cvr_1day'});
df_X_ct8_1day = df_X_ct8_1day.set_index(['item_category', 'time']);

df_X_ct9_1day = df_X_ct9_today.reset_index(level=[0, 1]);
df_X_ct9_1day.time = df_X_ct9_1day.time.add(np.timedelta64(1, 'D'));
df_X_ct9_1day = df_X_ct9_1day.rename(columns={'ct_buy_pr_today':'ct_buy_pr_1day'});
df_X_ct9_1day = df_X_ct9_1day.set_index(['item_category', 'time']);

df_X_ct10_1day = df_X_ct10_today.reset_index(level=[0, 1]);
df_X_ct10_1day.time = df_X_ct10_1day.time.add(np.timedelta64(1, 'D'));
df_X_ct10_1day = df_X_ct10_1day.rename(columns={'ct_ck_pr_today':'ct_ck_pr_1day'});
df_X_ct10_1day = df_X_ct10_1day.set_index(['item_category', 'time']);

df_X_ct11_1day = df_X_ct11_today.reset_index(level=[0, 1]);
df_X_ct11_1day.time = df_X_ct11_1day.time.add(np.timedelta64(1, 'D'));
df_X_ct11_1day = df_X_ct11_1day.rename(columns={'ct_cl_pr_today':'ct_cl_pr_1day'});
df_X_ct11_1day = df_X_ct11_1day.set_index(['item_category', 'time']);

df_X_ct12_1day = df_X_ct12_today.reset_index(level=[0, 1]);
df_X_ct12_1day.time = df_X_ct12_1day.time.add(np.timedelta64(1, 'D'));
df_X_ct12_1day = df_X_ct12_1day.rename(columns={'ct_ct_pr_today':'ct_ct_pr_1day'});
df_X_ct12_1day = df_X_ct12_1day.set_index(['item_category', 'time']);

#for 2 days before
df_X_ct1_2day = df_X_ct1_today.reset_index(level=[0, 1]);
df_X_ct1_2day.time = df_X_ct1_2day.time.add(np.timedelta64(2, 'D'));
df_X_ct1_2day = df_X_ct1_2day.rename(columns={'ct_buy_today':'ct_buy_2day'});
df_X_ct1_2day = df_X_ct1_2day.set_index(['item_category', 'time']);

df_X_ct2_2day = df_X_ct2_today.reset_index(level=[0, 1]);
df_X_ct2_2day.time = df_X_ct2_2day.time.add(np.timedelta64(2, 'D'));
df_X_ct2_2day = df_X_ct2_2day.rename(columns={'ct_click_today':'ct_click_2day'});
df_X_ct2_2day = df_X_ct2_2day.set_index(['item_category', 'time']);

df_X_ct3_2day = df_X_ct3_today.reset_index(level=[0, 1]);
df_X_ct3_2day.time = df_X_ct3_2day.time.add(np.timedelta64(2, 'D'));
df_X_ct3_2day = df_X_ct3_2day.rename(columns={'ct_collec_today':'ct_collec_2day'});
df_X_ct3_2day = df_X_ct3_2day.set_index(['item_category', 'time']);

df_X_ct4_2day = df_X_ct4_today.reset_index(level=[0, 1]);
df_X_ct4_2day.time = df_X_ct4_2day.time.add(np.timedelta64(2, 'D'));
df_X_ct4_2day = df_X_ct4_2day.rename(columns={'ct_cart_today':'ct_cart_2day'});
df_X_ct4_2day = df_X_ct4_2day.set_index(['item_category', 'time']);

df_X_ct5_2day = df_X_ct5_today.reset_index(level=[0, 1]);
df_X_ct5_2day.time = df_X_ct5_2day.time.add(np.timedelta64(2, 'D'));
df_X_ct5_2day = df_X_ct5_2day.rename(columns={'ct_bh_cvr_today':'ct_bh_cvr_2day'});
df_X_ct5_2day = df_X_ct5_2day.set_index(['item_category', 'time']);

df_X_ct6_2day = df_X_ct6_today.reset_index(level=[0, 1]);
df_X_ct6_2day.time = df_X_ct6_2day.time.add(np.timedelta64(2, 'D'));
df_X_ct6_2day = df_X_ct6_2day.rename(columns={'ct_ck_cvr_today':'ct_ck_cvr_2day'});
df_X_ct6_2day = df_X_ct6_2day.set_index(['item_category', 'time']);

df_X_ct7_2day = df_X_ct7_today.reset_index(level=[0, 1]);
df_X_ct7_2day.time = df_X_ct7_2day.time.add(np.timedelta64(2, 'D'));
df_X_ct7_2day = df_X_ct7_2day.rename(columns={'ct_cl_cvr_today':'ct_cl_cvr_2day'});
df_X_ct7_2day = df_X_ct7_2day.set_index(['item_category', 'time']);

df_X_ct8_2day = df_X_ct8_today.reset_index(level=[0, 1]);
df_X_ct8_2day.time = df_X_ct8_2day.time.add(np.timedelta64(2, 'D'));
df_X_ct8_2day = df_X_ct8_2day.rename(columns={'ct_ct_cvr_today':'ct_ct_cvr_2day'});
df_X_ct8_2day = df_X_ct8_2day.set_index(['item_category', 'time']);

df_X_ct9_2day = df_X_ct9_today.reset_index(level=[0, 1]);
df_X_ct9_2day.time = df_X_ct9_2day.time.add(np.timedelta64(2, 'D'));
df_X_ct9_2day = df_X_ct9_2day.rename(columns={'ct_buy_pr_today':'ct_buy_pr_2day'});
df_X_ct9_2day = df_X_ct9_2day.set_index(['item_category', 'time']);

df_X_ct10_2day = df_X_ct10_today.reset_index(level=[0, 1]);
df_X_ct10_2day.time = df_X_ct10_2day.time.add(np.timedelta64(2, 'D'));
df_X_ct10_2day = df_X_ct10_2day.rename(columns={'ct_ck_pr_today':'ct_ck_pr_2day'});
df_X_ct10_2day = df_X_ct10_2day.set_index(['item_category', 'time']);

df_X_ct11_2day = df_X_ct11_today.reset_index(level=[0, 1]);
df_X_ct11_2day.time = df_X_ct11_2day.time.add(np.timedelta64(2, 'D'));
df_X_ct11_2day = df_X_ct11_2day.rename(columns={'ct_cl_pr_today':'ct_cl_pr_2day'});
df_X_ct11_2day = df_X_ct11_2day.set_index(['item_category', 'time']);

df_X_ct12_2day = df_X_ct12_today.reset_index(level=[0, 1]);
df_X_ct12_2day.time = df_X_ct12_2day.time.add(np.timedelta64(2, 'D'));
df_X_ct12_2day = df_X_ct12_2day.rename(columns={'ct_ct_pr_today':'ct_ct_pr_2day'});
df_X_ct12_2day = df_X_ct12_2day.set_index(['item_category', 'time']);

#for 3-10 days before
df_X_ct_day = [];
for i in range(3, 11):
	df_X_ct_iday = df_clean.loc[:, ['item_id', 'item_category', 'time', 'behavior_type']];
	df_X_ct_iday.time = df_X_ct_iday.time.add(np.timedelta64(int(i), 'D'));
	df_X_ctn = [];
	df_X_ctn.append(df_X_ct_iday.loc[df_X_ct_iday.behavior_type==4].groupby(['item_category', 'time']).size().reset_index(name='ct_buy_' + str(i) + 'day'));
	df_X_ctn.append(df_X_ct_iday.loc[df_X_ct_iday.behavior_type==1].groupby(['item_category', 'time']).size().reset_index(name='ct_click_' + str(i) + 'day'));
	df_X_ctn.append(df_X_ct_iday.loc[df_X_ct_iday.behavior_type==2].groupby(['item_category', 'time']).size().reset_index(name='ct_collec_' + str(i) + 'day'));
	df_X_ctn.append(df_X_ct_iday.loc[df_X_ct_iday.behavior_type==3].groupby(['item_category', 'time']).size().reset_index(name='ct_cart_' + str(i) + 'day'));
	df_X_ct_day.append(df_X_ctn);
df_X_ct1_310day = df_X_ct_day[0][0].rename(columns={'ct_buy_3day':'ct_buy_310day'}).set_index(['item_category', 'time']);
df_X_ct2_310day = df_X_ct_day[0][1].rename(columns={'ct_click_3day':'ct_click_310day'}).set_index(['item_category', 'time']);
df_X_ct3_310day = df_X_ct_day[0][2].rename(columns={'ct_collec_3day':'ct_collec_310day'}).set_index(['item_category', 'time']);
df_X_ct4_310day = df_X_ct_day[0][3].rename(columns={'ct_cart_3day':'ct_cart_310day'}).set_index(['item_category', 'time']);
df_X_ct1_310day.ct_buy_310day = 0;
df_X_ct2_310day.ct_click_310day = 0;
df_X_ct3_310day.ct_collec_310day = 0;
df_X_ct4_310day.ct_cart_310day = 0;
for i in range(0, 8):
    df_X_ct1_310day = df_X_ct1_310day.join(df_X_ct_day[i][0].set_index(['item_category', 'time']), how='outer');
    df_X_ct2_310day = df_X_ct2_310day.join(df_X_ct_day[i][1].set_index(['item_category', 'time']), how='outer');
    df_X_ct3_310day = df_X_ct3_310day.join(df_X_ct_day[i][2].set_index(['item_category', 'time']), how='outer');
    df_X_ct4_310day = df_X_ct4_310day.join(df_X_ct_day[i][3].set_index(['item_category', 'time']), how='outer');
df_X_ct1_310day.fillna(0, inplace=True);
df_X_ct2_310day.fillna(0, inplace=True);
df_X_ct3_310day.fillna(0, inplace=True);
df_X_ct4_310day.fillna(0, inplace=True);
for i in range(1, 9):
    df_X_ct1_310day.ix[:, 0] = df_X_ct1_310day.ix[:, 0] + df_X_ct1_310day.ix[:, i] * (0.65 ** (np.log((i+2)**2 + 1))); #time decay
    df_X_ct2_310day.ix[:, 0] = df_X_ct2_310day.ix[:, 0] + df_X_ct2_310day.ix[:, i] * (0.65 ** (np.log((i+2)**2 + 1))); 
    df_X_ct3_310day.ix[:, 0] = df_X_ct3_310day.ix[:, 0] + df_X_ct3_310day.ix[:, i] * (0.65 ** (np.log((i+2)**2 + 1))); 
    df_X_ct4_310day.ix[:, 0] = df_X_ct4_310day.ix[:, 0] + df_X_ct4_310day.ix[:, i] * (0.65 ** (np.log((i+2)**2 + 1))); 
df_X_ct1_310day.drop(df_X_ct1_310day.columns[[1, 2, 3, 4, 5, 6, 7, 8]], axis=1, inplace=True);
df_X_ct2_310day.drop(df_X_ct2_310day.columns[[1, 2, 3, 4, 5, 6, 7, 8]], axis=1, inplace=True);
df_X_ct3_310day.drop(df_X_ct3_310day.columns[[1, 2, 3, 4, 5, 6, 7, 8]], axis=1, inplace=True);
df_X_ct4_310day.drop(df_X_ct4_310day.columns[[1, 2, 3, 4, 5, 6, 7, 8]], axis=1, inplace=True);

df_X_ct5_310day = df_X_ct1_310day.rename(columns={'ct_buy_310day':'ct_bh_cvr_310day'});
df_X_ct5_310day = df_X_ct5_310day.join(df_X_ct1_310day, how='outer').join(df_X_ct2_310day, how='outer').join(df_X_ct3_310day, how='outer').join(df_X_ct4_310day, how='outer');
df_X_ct5_310day.fillna(0, inplace=True);
df_X_ct5_310day.ct_bh_cvr_310day = df_X_ct5_310day.ct_buy_310day / (df_X_ct5_310day.ct_buy_310day + df_X_ct5_310day.ct_click_310day + df_X_ct5_310day.ct_collec_310day + df_X_ct5_310day.ct_cart_310day);
df_X_ct5_310day.drop(df_X_ct5_310day.columns[[1, 2, 3, 4]], axis=1, inplace=True);

df_X_ct6_310day = df_X_ct1_310day.rename(columns={'ct_buy_310day':'ct_ck_cvr_310day'});
df_X_ct6_310day = df_X_ct6_310day.join(df_X_ct1_310day, how='outer').join(df_X_ct2_310day, how='outer');
df_X_ct6_310day.fillna(0, inplace=True);
df_X_ct6_310day.ct_ck_cvr_310day = df_X_ct6_310day.ct_buy_310day / df_X_ct6_310day.ct_click_310day;
df_X_ct6_310day.replace(np.inf, 0, inplace=True);
df_X_ct6_310day.drop(df_X_ct6_310day.columns[[1, 2]], axis=1, inplace=True);

df_X_ct7_310day = df_X_ct1_310day.rename(columns={'ct_buy_310day':'ct_cl_cvr_310day'});
df_X_ct7_310day = df_X_ct7_310day.join(df_X_ct1_310day, how='outer').join(df_X_ct3_310day, how='outer');
df_X_ct7_310day.fillna(0, inplace=True);
df_X_ct7_310day.ct_cl_cvr_310day = df_X_ct7_310day.ct_buy_310day / df_X_ct7_310day.ct_collec_310day;
df_X_ct7_310day.replace(np.inf, 0, inplace=True);
df_X_ct7_310day.drop(df_X_ct7_310day.columns[[1, 2]], axis=1, inplace=True);

df_X_ct8_310day = df_X_ct1_310day.rename(columns={'ct_buy_310day':'ct_ct_cvr_310day'});
df_X_ct8_310day = df_X_ct8_310day.join(df_X_ct1_310day, how='outer').join(df_X_ct4_310day, how='outer');
df_X_ct8_310day.fillna(0, inplace=True);
df_X_ct8_310day.ct_ct_cvr_310day = df_X_ct8_310day.ct_buy_310day / df_X_ct8_310day.ct_cart_310day;
df_X_ct8_310day.replace(np.inf, 0, inplace=True);
df_X_ct8_310day.drop(df_X_ct8_310day.columns[[1, 2]], axis=1, inplace=True);

df_X_ct9_310day = df_X_ct1_310day.rename(columns={'ct_buy_310day':'ct_buy_pr_310day'});
df_X_ct9_310day = df_X_ct9_310day.join(df_X_ct1_310day, how='outer').join(df_X_c1, how='left');
df_X_ct9_310day.ct_buy_pr_310day = df_X_ct9_310day.ct_buy_310day / df_X_ct9_310day.cg_buy;
df_X_ct9_310day.drop(df_X_ct9_310day.columns[[1, 2]],axis=1, inplace=True);

df_X_ct10_310day = df_X_ct2_310day.rename(columns={'ct_click_310day':'ct_ck_pr_310day'});
df_X_ct10_310day = df_X_ct10_310day.join(df_X_ct2_310day, how='outer').join(df_X_c2, how='left');
df_X_ct10_310day.ct_ck_pr_310day = df_X_ct10_310day.ct_click_310day / df_X_ct10_310day.cg_click;
df_X_ct10_310day.drop(df_X_ct10_310day.columns[[1, 2]],axis=1, inplace=True);

df_X_ct11_310day = df_X_ct3_310day.rename(columns={'ct_collec_310day':'ct_cl_pr_310day'});
df_X_ct11_310day = df_X_ct11_310day.join(df_X_ct3_310day, how='outer').join(df_X_c3, how='left');
df_X_ct11_310day.ct_cl_pr_310day = df_X_ct11_310day.ct_collec_310day / df_X_ct11_310day.cg_collec;
df_X_ct11_310day.drop(df_X_ct11_310day.columns[[1, 2]],axis=1, inplace=True);

df_X_ct12_310day = df_X_ct4_310day.rename(columns={'ct_cart_310day':'ct_ct_pr_310day'});
df_X_ct12_310day = df_X_ct12_310day.join(df_X_ct4_310day, how='outer').join(df_X_c4, how='left');
df_X_ct12_310day.ct_ct_pr_310day = df_X_ct12_310day.ct_cart_310day / df_X_ct12_310day.cg_cart;
df_X_ct12_310day.drop(df_X_ct12_310day.columns[[1, 2]],axis=1, inplace=True);

#for 0-10 days before
df_X_ct_day = [];
for i in range(0, 11):
	df_X_ct_iday = df_clean.loc[:, ['user_id', 'item_category', 'time', 'behavior_type']];
	df_X_ct_iday.time = df_X_ct_iday.time.add(np.timedelta64(int(i), 'D'));
	df_X_ct_day.append(df_X_ct_iday.loc[df_X_ct_iday.behavior_type==4].groupby(['user_id', 'item_category', 'time']).size().reset_index(name='ct_buy_' + str(i) + 'day'));
df_X_ct1_010day = df_X_ct_day[0].rename(columns={'ct_buy_0day':'ct_buy_dp_010day'}).set_index(['user_id', 'item_category', 'time']);
df_X_ct1_010day.ct_buy_dp_010day = 0;
for i in range(0, 11):
	df_X_ct1_010day = df_X_ct1_010day.join(df_X_ct_day[i].set_index(['user_id', 'item_category', 'time']), how='outer');
df_X_ct1_010day.fillna(0, inplace=True);
df_X_ct1_010day = (df_X_ct1_010day!=0).astype(int);
for i in range(1, 12):
	df_X_ct1_010day.ix[:, 0] = df_X_ct1_010day.ix[:, 0] + df_X_ct1_010day.ix[:, i];
df_X_ct1_010day.drop(df_X_ct1_010day.columns[[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]], axis=1, inplace=True);
df_X_ct1_010day.ct_buy_dp_010day = (df_X_ct1_010day.ct_buy_dp_010day!=1);
df_X_ct1_010day.ct_buy_dp_010day = df_X_ct1_010day.ct_buy_dp_010day.astype(int);
df_X_ct1_010day = df_X_ct1_010day.reset_index(level=[0, 1, 2]);
df_X_ct1_010day.drop(df_X_ct1_010day.columns[[0]], axis=1, inplace=True);
df_X_ct1_010day = df_X_ct1_010day.groupby(['item_category', 'time']).mean();

df_X_ct_day = [];
for i in range(0, 11):
	df_X_ct_iday = df_clean.loc[:, ['user_id', 'item_category', 'time', 'behavior_type']];
	df_X_ct_iday.time = df_X_ct_iday.time.add(np.timedelta64(int(i), 'D'));
	df_X_ct_day.append(df_X_ct_iday.loc[df_X_ct_iday.behavior_type==4].groupby(['item_category', 'time']).size().reset_index(name='ct_buy_' + str(i) + 'day'));
df_X_ct2_010day = df_X_ct_day[0].rename(columns={'ct_buy_0day':'ct_co_buy_pre_010day'}).set_index(['item_category', 'time']);
df_X_ct2_010day.ct_co_buy_pre_010day = 0;
for i in range(0, 11):
	df_X_ct2_010day = df_X_ct2_010day.join(df_X_ct_day[i].set_index(['item_category', 'time']), how='outer');
df_X_ct2_010day.fillna(0, inplace=True);
df_X_ct2_010day = (df_X_ct2_010day!=0).astype(int);
for i in range(1, 12):
	df_X_ct2_010day.ix[:, 0] = df_X_ct2_010day.ix[:, 0] + df_X_ct2_010day.ix[:, i];
for i in range(1, 11):
	df_X_ct2_010day.ix[:, i] = (df_X_ct2_010day.ix[:, i].astype(bool) & df_X_ct2_010day.ix[:, i+1].astype(bool));
	df_X_ct2_010day.ix[:, i] = df_X_ct2_010day.ix[:, i].astype(int);
for i in range(2, 11):
	df_X_ct2_010day.ix[:, 1] = df_X_ct2_010day.ix[:, 1] + df_X_ct2_010day.ix[:, i];
df_X_ct2_010day.ix[:, 0] = df_X_ct2_010day.ix[:, 1] / df_X_ct2_010day.ix[:, 0];
df_X_ct2_010day.drop(df_X_ct2_010day.columns[[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]], axis=1, inplace=True);

#for the while time
df_X_ct1 = df_clean.loc[:, ['user_id', 'item_category', 'time', 'behavior_type']].groupby(['item_category', 'time']).size().reset_index(name='ct_bh');
df_X_ct1.drop(['ct_bh'], axis=1, inplace=True);
df_X_ct1m = df_clean.loc[df_clean.behavior_type==4].groupby(['item_category', 'time']).size().reset_index(name='ct_buy');
df_X_ct1m.drop(['ct_buy'], axis=1, inplace=True);
df_X_ct1 = df_X_ct1.join(df_X_ct1m.set_index(['item_category']), on='item_category', rsuffix='_m', how='left');
df_X_ct1 = df_X_ct1.set_index(['item_category', 'time'], drop=False);
df_X_ct1.time_m = df_X_ct1.time - df_X_ct1.time_m;
df_X_ct1 = df_X_ct1.loc[df_X_ct1.time_m>np.timedelta64(0, 'D')];
df_X_ct1 = df_X_ct1.groupby(['item_category', 'time']).min();
df_X_ct1 = df_X_ct1.rename(columns={'time_m':'ct_buy_lasttime'});
df_X_ct1.ct_buy_lasttime = df_X_ct1.ct_buy_lasttime.dt.days;

df_X_ct2 = df_clean.loc[:, ['user_id', 'item_category', 'time', 'behavior_type']].groupby(['item_category', 'time']).size().reset_index(name='ct_bh');
df_X_ct2.drop(['ct_bh'], axis=1, inplace=True);
df_X_ct2m = df_X_ct2.loc[:, ['item_category', 'time']];
df_X_ct2m.index = df_X_ct2m.index + 1;
df_X_ct2 = df_X_ct2.join(df_X_ct2m, rsuffix='_m', how='left');
df_X_ct2 = df_X_ct2.set_index(['item_category', 'time'], drop=False);
df_X_ct2.item_category = df_X_ct2.item_category - df_X_ct2.item_category_m;
df_X_ct2.time = df_X_ct2.time - df_X_ct2.time_m;
df_X_ct2 = df_X_ct2.loc[df_X_ct2.item_category==0];
df_X_ct2 = df_X_ct2.rename(columns={'time':'ct_bh_lasttime'});
df_X_ct2.ct_bh_lasttime = df_X_ct2.ct_bh_lasttime.dt.days;
df_X_ct2.drop(['item_category', 'item_category_m', 'time_m'], axis=1, inplace=True);


#for today
df_X_it1_today = df_clean.loc[df_clean.behavior_type==4].groupby(['item_id', 'time']).size().reset_index(name='it_buy_today').set_index(['item_id', 'time']);
df_X_it2_today = df_clean.loc[df_clean.behavior_type==1].groupby(['item_id', 'time']).size().reset_index(name='it_click_today').set_index(['item_id', 'time']);
df_X_it3_today = df_clean.loc[df_clean.behavior_type==2].groupby(['item_id', 'time']).size().reset_index(name='it_collec_today').set_index(['item_id', 'time']);
df_X_it4_today = df_clean.loc[df_clean.behavior_type==3].groupby(['item_id', 'time']).size().reset_index(name='it_cart_today').set_index(['item_id', 'time']);

df_X_it5_today = df_X_it1_today.rename(columns={'it_buy_today':'it_bh_cvr_today'});
df_X_it5_today = df_X_it5_today.join(df_X_it1_today, how='outer').join(df_X_it2_today, how='outer').join(df_X_it3_today, how='outer').join(df_X_it4_today, how='outer');
df_X_it5_today.fillna(0, inplace=True);
df_X_it5_today.it_bh_cvr_today = df_X_it5_today.it_buy_today / (df_X_it5_today.it_buy_today + df_X_it5_today.it_click_today + df_X_it5_today.it_collec_today + df_X_it5_today.it_cart_today);
df_X_it5_today.drop(['it_buy_today', 'it_click_today', 'it_collec_today', 'it_cart_today'], axis=1, inplace=True);

df_X_it6_today = df_X_it1_today.rename(columns={'it_buy_today':'it_ck_cvr_today'});
df_X_it6_today = df_X_it6_today.join(df_X_it1_today, how='outer').join(df_X_it2_today, how='outer');
df_X_it6_today.fillna(0, inplace=True);
df_X_it6_today.it_ck_cvr_today = df_X_it6_today.it_buy_today / df_X_it6_today.it_click_today;
df_X_it6_today.replace(np.inf, 0, inplace=True);
df_X_it6_today.drop(['it_buy_today', 'it_click_today'], axis=1, inplace=True);

df_X_it7_today = df_X_it1_today.rename(columns={'it_buy_today':'it_cl_cvr_today'});
df_X_it7_today = df_X_it7_today.join(df_X_it1_today, how='outer').join(df_X_it3_today, how='outer');
df_X_it7_today.fillna(0, inplace=True);
df_X_it7_today.it_cl_cvr_today = df_X_it7_today.it_buy_today / df_X_it7_today.it_collec_today;
df_X_it7_today.replace(np.inf, 0, inplace=True);
df_X_it7_today.drop(['it_buy_today', 'it_collec_today'], axis=1, inplace=True);

df_X_it8_today = df_X_it1_today.rename(columns={'it_buy_today':'it_ct_cvr_today'});
df_X_it8_today = df_X_it8_today.join(df_X_it1_today, how='outer').join(df_X_it4_today, how='outer');
df_X_it8_today.fillna(0, inplace=True);
df_X_it8_today.it_ct_cvr_today = df_X_it8_today.it_buy_today / df_X_it8_today.it_cart_today;
df_X_it8_today.replace(np.inf, 0, inplace=True);
df_X_it8_today.drop(['it_buy_today', 'it_cart_today'], axis=1, inplace=True);

df_X_it_day = [];
for i in range(0, 11):
	df_X_it_iday = df_clean.loc[:, ['user_id', 'item_id', 'time', 'behavior_type']];
	df_X_it_iday.time = df_X_it_iday.time.add(np.timedelta64(int(i), 'D'));
	df_X_itn = [];
	df_X_itn.append(df_X_it_iday.loc[df_X_it_iday.behavior_type==4].groupby(['item_id', 'time']).size().reset_index(name='it_buy_' + str(i) + 'day'));
	df_X_itn.append(df_X_it_iday.loc[df_X_it_iday.behavior_type==1].groupby(['item_id', 'time']).size().reset_index(name='it_click_' + str(i) + 'day'));
	df_X_itn.append(df_X_it_iday.loc[df_X_it_iday.behavior_type==2].groupby(['item_id', 'time']).size().reset_index(name='it_collec_' + str(i) + 'day'));
	df_X_itn.append(df_X_it_iday.loc[df_X_it_iday.behavior_type==3].groupby(['item_id', 'time']).size().reset_index(name='it_cart_' + str(i) + 'day'));
	df_X_it_day.append(df_X_itn);
df_X_it1s_010day = df_X_it_day[0][0].rename(columns={'it_buy_0day':'it_buy_010day'}).set_index(['item_id', 'time']);
df_X_it2s_010day = df_X_it_day[0][1].rename(columns={'it_click_0day':'it_click_010day'}).set_index(['item_id', 'time']);
df_X_it3s_010day = df_X_it_day[0][2].rename(columns={'it_collec_0day':'it_collec_010day'}).set_index(['item_id', 'time']);
df_X_it4s_010day = df_X_it_day[0][3].rename(columns={'it_cart_0day':'it_cart_010day'}).set_index(['item_id', 'time']);
df_X_it1s_010day.it_buy_010day = 0;
df_X_it2s_010day.it_click_010day = 0;
df_X_it3s_010day.it_collec_010day = 0;
df_X_it4s_010day.it_cart_010day = 0;
for i in range(0, 11):
	df_X_it1s_010day = df_X_it1s_010day.join(df_X_it_day[i][0].set_index(['item_id', 'time']), how='outer');
	df_X_it2s_010day = df_X_it2s_010day.join(df_X_it_day[i][1].set_index(['item_id', 'time']), how='outer');
	df_X_it3s_010day = df_X_it3s_010day.join(df_X_it_day[i][2].set_index(['item_id', 'time']), how='outer');
	df_X_it4s_010day = df_X_it4s_010day.join(df_X_it_day[i][3].set_index(['item_id', 'time']), how='outer');
df_X_it1s_010day.fillna(0, inplace=True);
df_X_it2s_010day.fillna(0, inplace=True);
df_X_it3s_010day.fillna(0, inplace=True);
df_X_it4s_010day.fillna(0, inplace=True);
for i in range(1, 12):
	df_X_it1s_010day.ix[:, 0] = df_X_it1s_010day.ix[:, 0] + df_X_it1s_010day.ix[:, i] * (0.65 ** (np.log((i-1)**2 + 1)));
	df_X_it2s_010day.ix[:, 0] = df_X_it2s_010day.ix[:, 0] + df_X_it2s_010day.ix[:, i] * (0.65 ** (np.log((i-1)**2 + 1)));
	df_X_it3s_010day.ix[:, 0] = df_X_it3s_010day.ix[:, 0] + df_X_it3s_010day.ix[:, i] * (0.65 ** (np.log((i-1)**2 + 1)));
	df_X_it4s_010day.ix[:, 0] = df_X_it4s_010day.ix[:, 0] + df_X_it4s_010day.ix[:, i] * (0.65 ** (np.log((i-1)**2 + 1)));
df_X_it1s_010day.drop(df_X_it1s_010day.columns[[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]], axis=1, inplace=True);
df_X_it2s_010day.drop(df_X_it2s_010day.columns[[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]], axis=1, inplace=True);
df_X_it3s_010day.drop(df_X_it3s_010day.columns[[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]], axis=1, inplace=True);
df_X_it4s_010day.drop(df_X_it4s_010day.columns[[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]], axis=1, inplace=True);

df_X_it9_today = df_X_it1_today.rename(columns={'it_buy_today':'it_buy_pr_today'});
df_X_it9_today = df_X_it9_today.join(df_X_it1_today, how='outer').join(df_X_it1s_010day, how='left');
df_X_it9_today.fillna(0, inplace=True);
df_X_it9_today.it_buy_pr_today = df_X_it9_today.it_buy_today / df_X_it9_today.it_buy_010day;
df_X_it9_today.drop(df_X_it9_today.columns[[1, 2]],axis=1, inplace=True);

df_X_it10_today = df_X_it2_today.rename(columns={'it_click_today':'it_ck_pr_today'});
df_X_it10_today = df_X_it10_today.join(df_X_it2_today, how='outer').join(df_X_it2s_010day, how='left');
df_X_it10_today.fillna(0, inplace=True);
df_X_it10_today.it_ck_pr_today = df_X_it10_today.it_click_today / df_X_it10_today.it_click_010day;
df_X_it10_today.drop(df_X_it10_today.columns[[1, 2]],axis=1, inplace=True);

df_X_it11_today = df_X_it3_today.rename(columns={'it_collec_today':'it_cl_pr_today'});
df_X_it11_today = df_X_it11_today.join(df_X_it3_today, how='outer').join(df_X_it3s_010day, how='left');
df_X_it11_today.fillna(0, inplace=True);
df_X_it11_today.it_cl_pr_today = df_X_it11_today.it_collec_today / df_X_it11_today.it_collec_010day;
df_X_it11_today.drop(df_X_it11_today.columns[[1, 2]],axis=1, inplace=True);

df_X_it12_today = df_X_it4_today.rename(columns={'it_cart_today':'it_ct_pr_today'});
df_X_it12_today = df_X_it12_today.join(df_X_it4_today, how='outer').join(df_X_it4s_010day, how='left');
df_X_it12_today.fillna(0, inplace=True);
df_X_it12_today.it_ct_pr_today = df_X_it12_today.it_cart_today / df_X_it12_today.it_cart_010day;
df_X_it12_today.drop(df_X_it12_today.columns[[1, 2]],axis=1, inplace=True);

#for one day before
df_X_it1_1day = df_X_it1_today.reset_index(level=[0, 1]);
df_X_it1_1day.time = df_X_it1_1day.time.add(np.timedelta64(1, 'D'));
df_X_it1_1day = df_X_it1_1day.rename(columns={'it_buy_today':'it_buy_1day'});
df_X_it1_1day = df_X_it1_1day.set_index(['item_id', 'time']);

df_X_it2_1day = df_X_it2_today.reset_index(level=[0, 1]);
df_X_it2_1day.time = df_X_it2_1day.time.add(np.timedelta64(1, 'D'));
df_X_it2_1day = df_X_it2_1day.rename(columns={'it_click_today':'it_click_1day'});
df_X_it2_1day = df_X_it2_1day.set_index(['item_id', 'time']);

df_X_it3_1day = df_X_it3_today.reset_index(level=[0, 1]);
df_X_it3_1day.time = df_X_it3_1day.time.add(np.timedelta64(1, 'D'));
df_X_it3_1day = df_X_it3_1day.rename(columns={'it_collec_today':'it_collec_1day'});
df_X_it3_1day = df_X_it3_1day.set_index(['item_id', 'time']);

df_X_it4_1day = df_X_it4_today.reset_index(level=[0, 1]);
df_X_it4_1day.time = df_X_it4_1day.time.add(np.timedelta64(1, 'D'));
df_X_it4_1day = df_X_it4_1day.rename(columns={'it_cart_today':'it_cart_1day'});
df_X_it4_1day = df_X_it4_1day.set_index(['item_id', 'time']);

df_X_it5_1day = df_X_it5_today.reset_index(level=[0, 1]);
df_X_it5_1day.time = df_X_it5_1day.time.add(np.timedelta64(1, 'D'));
df_X_it5_1day = df_X_it5_1day.rename(columns={'it_bh_cvr_today':'it_bh_cvr_1day'});
df_X_it5_1day = df_X_it5_1day.set_index(['item_id', 'time']);

df_X_it6_1day = df_X_it6_today.reset_index(level=[0, 1]);
df_X_it6_1day.time = df_X_it6_1day.time.add(np.timedelta64(1, 'D'));
df_X_it6_1day = df_X_it6_1day.rename(columns={'it_ck_cvr_today':'it_ck_cvr_1day'});
df_X_it6_1day = df_X_it6_1day.set_index(['item_id', 'time']);

df_X_it7_1day = df_X_it7_today.reset_index(level=[0, 1]);
df_X_it7_1day.time = df_X_it7_1day.time.add(np.timedelta64(1, 'D'));
df_X_it7_1day = df_X_it7_1day.rename(columns={'it_cl_cvr_today':'it_cl_cvr_1day'});
df_X_it7_1day = df_X_it7_1day.set_index(['item_id', 'time']);

df_X_it8_1day = df_X_it8_today.reset_index(level=[0, 1]);
df_X_it8_1day.time = df_X_it8_1day.time.add(np.timedelta64(1, 'D'));
df_X_it8_1day = df_X_it8_1day.rename(columns={'it_ct_cvr_today':'it_ct_cvr_1day'});
df_X_it8_1day = df_X_it8_1day.set_index(['item_id', 'time']);

df_X_it9_1day = df_X_it9_today.reset_index(level=[0, 1]);
df_X_it9_1day.time = df_X_it9_1day.time.add(np.timedelta64(1, 'D'));
df_X_it9_1day = df_X_it9_1day.rename(columns={'it_buy_pr_today':'it_buy_pr_1day'});
df_X_it9_1day = df_X_it9_1day.set_index(['item_id', 'time']);

df_X_it10_1day = df_X_it10_today.reset_index(level=[0, 1]);
df_X_it10_1day.time = df_X_it10_1day.time.add(np.timedelta64(1, 'D'));
df_X_it10_1day = df_X_it10_1day.rename(columns={'it_ck_pr_today':'it_ck_pr_1day'});
df_X_it10_1day = df_X_it10_1day.set_index(['item_id', 'time']);

df_X_it11_1day = df_X_it11_today.reset_index(level=[0, 1]);
df_X_it11_1day.time = df_X_it11_1day.time.add(np.timedelta64(1, 'D'));
df_X_it11_1day = df_X_it11_1day.rename(columns={'it_cl_pr_today':'it_cl_pr_1day'});
df_X_it11_1day = df_X_it11_1day.set_index(['item_id', 'time']);

df_X_it12_1day = df_X_it12_today.reset_index(level=[0, 1]);
df_X_it12_1day.time = df_X_it12_1day.time.add(np.timedelta64(1, 'D'));
df_X_it12_1day = df_X_it12_1day.rename(columns={'it_ct_pr_today':'it_ct_pr_1day'});
df_X_it12_1day = df_X_it12_1day.set_index(['item_id', 'time']);

#for 2 days before
df_X_it1_2day = df_X_it1_today.reset_index(level=[0, 1]);
df_X_it1_2day.time = df_X_it1_2day.time.add(np.timedelta64(2, 'D'));
df_X_it1_2day = df_X_it1_2day.rename(columns={'it_buy_today':'it_buy_2day'});
df_X_it1_2day = df_X_it1_2day.set_index(['item_id', 'time']);

df_X_it2_2day = df_X_it2_today.reset_index(level=[0, 1]);
df_X_it2_2day.time = df_X_it2_2day.time.add(np.timedelta64(2, 'D'));
df_X_it2_2day = df_X_it2_2day.rename(columns={'it_click_today':'it_click_2day'});
df_X_it2_2day = df_X_it2_2day.set_index(['item_id', 'time']);

df_X_it3_2day = df_X_it3_today.reset_index(level=[0, 1]);
df_X_it3_2day.time = df_X_it3_2day.time.add(np.timedelta64(2, 'D'));
df_X_it3_2day = df_X_it3_2day.rename(columns={'it_collec_today':'it_collec_2day'});
df_X_it3_2day = df_X_it3_2day.set_index(['item_id', 'time']);

df_X_it4_2day = df_X_it4_today.reset_index(level=[0, 1]);
df_X_it4_2day.time = df_X_it4_2day.time.add(np.timedelta64(2, 'D'));
df_X_it4_2day = df_X_it4_2day.rename(columns={'it_cart_today':'it_cart_2day'});
df_X_it4_2day = df_X_it4_2day.set_index(['item_id', 'time']);

df_X_it5_2day = df_X_it5_today.reset_index(level=[0, 1]);
df_X_it5_2day.time = df_X_it5_2day.time.add(np.timedelta64(2, 'D'));
df_X_it5_2day = df_X_it5_2day.rename(columns={'it_bh_cvr_today':'it_bh_cvr_2day'});
df_X_it5_2day = df_X_it5_2day.set_index(['item_id', 'time']);

df_X_it6_2day = df_X_it6_today.reset_index(level=[0, 1]);
df_X_it6_2day.time = df_X_it6_2day.time.add(np.timedelta64(2, 'D'));
df_X_it6_2day = df_X_it6_2day.rename(columns={'it_ck_cvr_today':'it_ck_cvr_2day'});
df_X_it6_2day = df_X_it6_2day.set_index(['item_id', 'time']);

df_X_it7_2day = df_X_it7_today.reset_index(level=[0, 1]);
df_X_it7_2day.time = df_X_it7_2day.time.add(np.timedelta64(2, 'D'));
df_X_it7_2day = df_X_it7_2day.rename(columns={'it_cl_cvr_today':'it_cl_cvr_2day'});
df_X_it7_2day = df_X_it7_2day.set_index(['item_id', 'time']);

df_X_it8_2day = df_X_it8_today.reset_index(level=[0, 1]);
df_X_it8_2day.time = df_X_it8_2day.time.add(np.timedelta64(2, 'D'));
df_X_it8_2day = df_X_it8_2day.rename(columns={'it_ct_cvr_today':'it_ct_cvr_2day'});
df_X_it8_2day = df_X_it8_2day.set_index(['item_id', 'time']);

df_X_it9_2day = df_X_it9_today.reset_index(level=[0, 1]);
df_X_it9_2day.time = df_X_it9_2day.time.add(np.timedelta64(2, 'D'));
df_X_it9_2day = df_X_it9_2day.rename(columns={'it_buy_pr_today':'it_buy_pr_2day'});
df_X_it9_2day = df_X_it9_2day.set_index(['item_id', 'time']);

df_X_it10_2day = df_X_it10_today.reset_index(level=[0, 1]);
df_X_it10_2day.time = df_X_it10_2day.time.add(np.timedelta64(2, 'D'));
df_X_it10_2day = df_X_it10_2day.rename(columns={'it_ck_pr_today':'it_ck_pr_2day'});
df_X_it10_2day = df_X_it10_2day.set_index(['item_id', 'time']);

df_X_it11_2day = df_X_it11_today.reset_index(level=[0, 1]);
df_X_it11_2day.time = df_X_it11_2day.time.add(np.timedelta64(2, 'D'));
df_X_it11_2day = df_X_it11_2day.rename(columns={'it_cl_pr_today':'it_cl_pr_2day'});
df_X_it11_2day = df_X_it11_2day.set_index(['item_id', 'time']);

df_X_it12_2day = df_X_it12_today.reset_index(level=[0, 1]);
df_X_it12_2day.time = df_X_it12_2day.time.add(np.timedelta64(2, 'D'));
df_X_it12_2day = df_X_it12_2day.rename(columns={'it_ct_pr_today':'it_ct_pr_2day'});
df_X_it12_2day = df_X_it12_2day.set_index(['item_id', 'time']);

#for 3-10 days before
df_X_it_day = [];
for i in range(3, 11):
	df_X_it_iday = df_clean.loc[:, ['user_id', 'item_id', 'time', 'behavior_type']];
	df_X_it_iday.time = df_X_it_iday.time.add(np.timedelta64(int(i), 'D'));
	df_X_itn = [];
	df_X_itn.append(df_X_it_iday.loc[df_X_it_iday.behavior_type==4].groupby(['item_id', 'time']).size().reset_index(name='it_buy_' + str(i) + 'day'));
	df_X_itn.append(df_X_it_iday.loc[df_X_it_iday.behavior_type==1].groupby(['item_id', 'time']).size().reset_index(name='it_click_' + str(i) + 'day'));
	df_X_itn.append(df_X_it_iday.loc[df_X_it_iday.behavior_type==2].groupby(['item_id', 'time']).size().reset_index(name='it_collec_' + str(i) + 'day'));
	df_X_itn.append(df_X_it_iday.loc[df_X_it_iday.behavior_type==3].groupby(['item_id', 'time']).size().reset_index(name='it_cart_' + str(i) + 'day'));
	df_X_it_day.append(df_X_itn);
df_X_it1_310day = df_X_it_day[0][0].rename(columns={'it_buy_3day':'it_buy_310day'}).set_index(['item_id', 'time']);
df_X_it2_310day = df_X_it_day[0][1].rename(columns={'it_click_3day':'it_click_310day'}).set_index(['item_id', 'time']);
df_X_it3_310day = df_X_it_day[0][2].rename(columns={'it_collec_3day':'it_collec_310day'}).set_index(['item_id', 'time']);
df_X_it4_310day = df_X_it_day[0][3].rename(columns={'it_cart_3day':'it_cart_310day'}).set_index(['item_id', 'time']);
df_X_it1_310day.it_buy_310day = 0;
df_X_it2_310day.it_click_310day = 0;
df_X_it3_310day.it_collec_310day = 0;
df_X_it4_310day.it_cart_310day = 0;
for i in range(0, 8):
    df_X_it1_310day = df_X_it1_310day.join(df_X_it_day[i][0].set_index(['item_id', 'time']), how='outer');
    df_X_it2_310day = df_X_it2_310day.join(df_X_it_day[i][1].set_index(['item_id', 'time']), how='outer');
    df_X_it3_310day = df_X_it3_310day.join(df_X_it_day[i][2].set_index(['item_id', 'time']), how='outer');
    df_X_it4_310day = df_X_it4_310day.join(df_X_it_day[i][3].set_index(['item_id', 'time']), how='outer');
df_X_it1_310day.fillna(0, inplace=True);
df_X_it2_310day.fillna(0, inplace=True);
df_X_it3_310day.fillna(0, inplace=True);
df_X_it4_310day.fillna(0, inplace=True);
for i in range(1, 9):
    df_X_it1_310day.ix[:, 0] = df_X_it1_310day.ix[:, 0] + df_X_it1_310day.ix[:, i] * (0.65 ** (np.log((i+2)**2 + 1))); #time decay
    df_X_it2_310day.ix[:, 0] = df_X_it2_310day.ix[:, 0] + df_X_it2_310day.ix[:, i] * (0.65 ** (np.log((i+2)**2 + 1))); 
    df_X_it3_310day.ix[:, 0] = df_X_it3_310day.ix[:, 0] + df_X_it3_310day.ix[:, i] * (0.65 ** (np.log((i+2)**2 + 1))); 
    df_X_it4_310day.ix[:, 0] = df_X_it4_310day.ix[:, 0] + df_X_it4_310day.ix[:, i] * (0.65 ** (np.log((i+2)**2 + 1))); 
df_X_it1_310day.drop(df_X_it1_310day.columns[[1, 2, 3, 4, 5, 6, 7, 8]], axis=1, inplace=True);
df_X_it2_310day.drop(df_X_it2_310day.columns[[1, 2, 3, 4, 5, 6, 7, 8]], axis=1, inplace=True);
df_X_it3_310day.drop(df_X_it3_310day.columns[[1, 2, 3, 4, 5, 6, 7, 8]], axis=1, inplace=True);
df_X_it4_310day.drop(df_X_it4_310day.columns[[1, 2, 3, 4, 5, 6, 7, 8]], axis=1, inplace=True);

df_X_it5_310day = df_X_it1_310day.rename(columns={'it_buy_310day':'it_bh_cvr_310day'});
df_X_it5_310day = df_X_it5_310day.join(df_X_it1_310day, how='outer').join(df_X_it2_310day, how='outer').join(df_X_it3_310day, how='outer').join(df_X_it4_310day, how='outer');
df_X_it5_310day.fillna(0, inplace=True);
df_X_it5_310day.it_bh_cvr_310day = df_X_it5_310day.it_buy_310day / (df_X_it5_310day.it_buy_310day + df_X_it5_310day.it_click_310day + df_X_it5_310day.it_collec_310day + df_X_it5_310day.it_cart_310day);
df_X_it5_310day.drop(df_X_it5_310day.columns[[1, 2, 3, 4]], axis=1, inplace=True);

df_X_it6_310day = df_X_it1_310day.rename(columns={'it_buy_310day':'it_ck_cvr_310day'});
df_X_it6_310day = df_X_it6_310day.join(df_X_it1_310day, how='outer').join(df_X_it2_310day, how='outer');
df_X_it6_310day.fillna(0, inplace=True);
df_X_it6_310day.it_ck_cvr_310day = df_X_it6_310day.it_buy_310day / df_X_it6_310day.it_click_310day;
df_X_it6_310day.replace(np.inf, 0, inplace=True);
df_X_it6_310day.drop(df_X_it6_310day.columns[[1, 2]], axis=1, inplace=True);

df_X_it7_310day = df_X_it1_310day.rename(columns={'it_buy_310day':'it_cl_cvr_310day'});
df_X_it7_310day = df_X_it7_310day.join(df_X_it1_310day, how='outer').join(df_X_it3_310day, how='outer');
df_X_it7_310day.fillna(0, inplace=True);
df_X_it7_310day.it_cl_cvr_310day = df_X_it7_310day.it_buy_310day / df_X_it7_310day.it_collec_310day;
df_X_it7_310day.replace(np.inf, 0, inplace=True);
df_X_it7_310day.drop(df_X_it7_310day.columns[[1, 2]], axis=1, inplace=True);

df_X_it8_310day = df_X_it1_310day.rename(columns={'it_buy_310day':'it_ct_cvr_310day'});
df_X_it8_310day = df_X_it8_310day.join(df_X_it1_310day, how='outer').join(df_X_it4_310day, how='outer');
df_X_it8_310day.fillna(0, inplace=True);
df_X_it8_310day.it_ct_cvr_310day = df_X_it8_310day.it_buy_310day / df_X_it8_310day.it_cart_310day;
df_X_it8_310day.replace(np.inf, 0, inplace=True);
df_X_it8_310day.drop(df_X_it8_310day.columns[[1, 2]], axis=1, inplace=True);

df_X_it9_310day = df_X_it1_310day.rename(columns={'it_buy_310day':'it_buy_pr_310day'});
df_X_it9_310day = df_X_it9_310day.join(df_X_it1_310day, how='outer').join(df_X_i1, how='left');
df_X_it9_310day.it_buy_pr_310day = df_X_it9_310day.it_buy_310day / df_X_it9_310day.it_buy;
df_X_it9_310day.drop(df_X_it9_310day.columns[[1, 2]],axis=1, inplace=True);

df_X_it10_310day = df_X_it2_310day.rename(columns={'it_click_310day':'it_ck_pr_310day'});
df_X_it10_310day = df_X_it10_310day.join(df_X_it2_310day, how='outer').join(df_X_i2, how='left');
df_X_it10_310day.it_ck_pr_310day = df_X_it10_310day.it_click_310day / df_X_it10_310day.it_click;
df_X_it10_310day.drop(df_X_it10_310day.columns[[1, 2]],axis=1, inplace=True);

df_X_it11_310day = df_X_it3_310day.rename(columns={'it_collec_310day':'it_cl_pr_310day'});
df_X_it11_310day = df_X_it11_310day.join(df_X_it3_310day, how='outer').join(df_X_i3, how='left');
df_X_it11_310day.it_cl_pr_310day = df_X_it11_310day.it_collec_310day / df_X_it11_310day.it_collec;
df_X_it11_310day.drop(df_X_it11_310day.columns[[1, 2]],axis=1, inplace=True);

df_X_it12_310day = df_X_it4_310day.rename(columns={'it_cart_310day':'it_ct_pr_310day'});
df_X_it12_310day = df_X_it12_310day.join(df_X_it4_310day, how='outer').join(df_X_i4, how='left');
df_X_it12_310day.it_ct_pr_310day = df_X_it12_310day.it_cart_310day / df_X_it12_310day.it_cart;
df_X_it12_310day.drop(df_X_it12_310day.columns[[1, 2]],axis=1, inplace=True);

#for 0-10 days before
df_X_it_day = [];
for i in range(0, 11):
	df_X_it_iday = df_clean.loc[:, ['user_id', 'item_id', 'time', 'behavior_type']];
	df_X_it_iday.time = df_X_it_iday.time.add(np.timedelta64(int(i), 'D'));
	df_X_it_day.append(df_X_it_iday.loc[df_X_it_iday.behavior_type==4].groupby(['user_id', 'item_id', 'time']).size().reset_index(name='it_buy_' + str(i) + 'day'));
df_X_it1_010day = df_X_it_day[0].rename(columns={'it_buy_0day':'it_buy_dp_010day'}).set_index(['user_id', 'item_id', 'time']);
df_X_it1_010day.it_buy_dp_010day = 0;
for i in range(0, 11):
	df_X_it1_010day = df_X_it1_010day.join(df_X_it_day[i].set_index(['user_id', 'item_id', 'time']), how='outer');
df_X_it1_010day.fillna(0, inplace=True);
df_X_it1_010day = (df_X_it1_010day!=0).astype(int);
for i in range(1, 12):
	df_X_it1_010day.ix[:, 0] = df_X_it1_010day.ix[:, 0] + df_X_it1_010day.ix[:, i];
df_X_it1_010day.drop(df_X_it1_010day.columns[[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]], axis=1, inplace=True);
df_X_it1_010day.it_buy_dp_010day = (df_X_it1_010day.it_buy_dp_010day!=1);
df_X_it1_010day.it_buy_dp_010day = df_X_it1_010day.it_buy_dp_010day.astype(int);
df_X_it1_010day = df_X_it1_010day.reset_index(level=[0, 1, 2]);
df_X_it1_010day.drop(df_X_it1_010day.columns[[0]], axis=1, inplace=True);
df_X_it1_010day = df_X_it1_010day.groupby(['item_id', 'time']).mean();

df_X_it_day = [];
for i in range(0, 11):
	df_X_it_iday = df_clean.loc[:, ['user_id', 'item_id', 'time', 'behavior_type']];
	df_X_it_iday.time = df_X_it_iday.time.add(np.timedelta64(int(i), 'D'));
	df_X_it_day.append(df_X_it_iday.loc[df_X_it_iday.behavior_type==4].groupby(['item_id', 'time']).size().reset_index(name='it_buy_' + str(i) + 'day'));
df_X_it2_010day = df_X_it_day[0].rename(columns={'it_buy_0day':'it_co_buy_pre_010day'}).set_index(['item_id', 'time']);
df_X_it2_010day.it_co_buy_pre_010day = 0;
for i in range(0, 11):
	df_X_it2_010day = df_X_it2_010day.join(df_X_it_day[i].set_index(['item_id', 'time']), how='outer');
df_X_it2_010day.fillna(0, inplace=True);
df_X_it2_010day = (df_X_it2_010day!=0).astype(int);
for i in range(1, 12):
	df_X_it2_010day.ix[:, 0] = df_X_it2_010day.ix[:, 0] + df_X_it2_010day.ix[:, i];
for i in range(1, 11):
	df_X_it2_010day.ix[:, i] = (df_X_it2_010day.ix[:, i].astype(bool) & df_X_it2_010day.ix[:, i+1].astype(bool));
	df_X_it2_010day.ix[:, i] = df_X_it2_010day.ix[:, i].astype(int);
for i in range(2, 11):
	df_X_it2_010day.ix[:, 1] = df_X_it2_010day.ix[:, 1] + df_X_it2_010day.ix[:, i];
df_X_it2_010day.ix[:, 0] = df_X_it2_010day.ix[:, 1] / df_X_it2_010day.ix[:, 0];
df_X_it2_010day.drop(df_X_it2_010day.columns[[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]], axis=1, inplace=True);

#for the while time
df_X_it1 = df_clean.loc[:, ['user_id', 'item_id', 'time', 'behavior_type']].groupby(['item_id', 'time']).size().reset_index(name='it_bh');
df_X_it1.drop(['it_bh'], axis=1, inplace=True);
df_X_it1m = df_clean.loc[df_clean.behavior_type==4].groupby(['item_id', 'time']).size().reset_index(name='it_buy');
df_X_it1m.drop(['it_buy'], axis=1, inplace=True);
df_X_it1 = df_X_it1.join(df_X_it1m.set_index(['item_id']), on='item_id', rsuffix='_m', how='left');
df_X_it1 = df_X_it1.set_index(['item_id', 'time'], drop=False);
df_X_it1.time_m = df_X_it1.time - df_X_it1.time_m;
df_X_it1 = df_X_it1.loc[df_X_it1.time_m>np.timedelta64(0, 'D')];
df_X_it1 = df_X_it1.groupby(['item_id', 'time']).min();
df_X_it1 = df_X_it1.rename(columns={'time_m':'it_buy_lasttime'});
df_X_it1.it_buy_lasttime = df_X_it1.it_buy_lasttime.dt.days;

df_X_it2 = df_clean.loc[:, ['user_id', 'item_id', 'time', 'behavior_type']].groupby(['item_id', 'time']).size().reset_index(name='it_bh');
df_X_it2.drop(['it_bh'], axis=1, inplace=True);
df_X_it2m = df_X_it2.loc[:, ['item_id', 'time']];
df_X_it2m.index = df_X_it2m.index + 1;
df_X_it2 = df_X_it2.join(df_X_it2m, rsuffix='_m', how='left');
df_X_it2 = df_X_it2.set_index(['item_id', 'time'], drop=False);
df_X_it2.item_id = df_X_it2.item_id - df_X_it2.item_id_m;
df_X_it2.time = df_X_it2.time - df_X_it2.time_m;
df_X_it2 = df_X_it2.loc[df_X_it2.item_id==0];
df_X_it2 = df_X_it2.rename(columns={'time':'it_bh_lasttime'});
df_X_it2.it_bh_lasttime = df_X_it2.it_bh_lasttime.dt.days;
df_X_it2.drop(['item_id', 'item_id_m', 'time_m'], axis=1, inplace=True);

##for user X item(category) X time
#for today
df_X_uct1_today = df_clean.loc[df_clean.behavior_type==4].groupby(['user_id', 'item_category', 'time']).size().reset_index(name='uct_buy_today').set_index(['user_id', 'item_category', 'time']);
df_X_uct2_today = df_clean.loc[df_clean.behavior_type==1].groupby(['user_id', 'item_category', 'time']).size().reset_index(name='uct_click_today').set_index(['user_id', 'item_category', 'time']);
df_X_uct3_today = df_clean.loc[df_clean.behavior_type==2].groupby(['user_id', 'item_category', 'time']).size().reset_index(name='uct_collec_today').set_index(['user_id', 'item_category', 'time']);
df_X_uct4_today = df_clean.loc[df_clean.behavior_type==3].groupby(['user_id', 'item_category', 'time']).size().reset_index(name='uct_cart_today').set_index(['user_id', 'item_category', 'time']);

#for one day before
df_X_uct1_1day = df_X_uct1_today.reset_index(level=[0, 1, 2]);
df_X_uct1_1day.time = df_X_uct1_1day.time.add(np.timedelta64(1, 'D'));
df_X_uct1_1day = df_X_uct1_1day.rename(columns={'uct_buy_today':'uct_buy_1day'});
df_X_uct1_1day = df_X_uct1_1day.set_index(['user_id', 'item_category', 'time']);

df_X_uct2_1day = df_X_uct2_today.reset_index(level=[0, 1, 2]);
df_X_uct2_1day.time = df_X_uct2_1day.time.add(np.timedelta64(1, 'D'));
df_X_uct2_1day = df_X_uct2_1day.rename(columns={'uct_click_today':'uct_click_1day'});
df_X_uct2_1day = df_X_uct2_1day.set_index(['user_id','item_category', 'time']);

df_X_uct3_1day = df_X_uct3_today.reset_index(level=[0, 1, 2]);
df_X_uct3_1day.time = df_X_uct3_1day.time.add(np.timedelta64(1, 'D'));
df_X_uct3_1day = df_X_uct3_1day.rename(columns={'uct_collec_today':'uct_collec_1day'});
df_X_uct3_1day = df_X_uct3_1day.set_index(['user_id', 'item_category', 'time']);

df_X_uct4_1day = df_X_uct4_today.reset_index(level=[0, 1, 2]);
df_X_uct4_1day.time = df_X_uct4_1day.time.add(np.timedelta64(1, 'D'));
df_X_uct4_1day = df_X_uct4_1day.rename(columns={'uct_cart_today':'uct_cart_1day'});
df_X_uct4_1day = df_X_uct4_1day.set_index(['user_id', 'item_category', 'time']);

#for 2 days before
df_X_uct1_2day = df_X_uct1_today.reset_index(level=[0, 1, 2]);
df_X_uct1_2day.time = df_X_uct1_2day.time.add(np.timedelta64(2, 'D'));
df_X_uct1_2day = df_X_uct1_2day.rename(columns={'uct_buy_today':'uct_buy_2day'});
df_X_uct1_2day = df_X_uct1_2day.set_index(['user_id', 'item_category', 'time']);

df_X_uct2_2day = df_X_uct2_today.reset_index(level=[0, 1, 2]);
df_X_uct2_2day.time = df_X_uct2_2day.time.add(np.timedelta64(2, 'D'));
df_X_uct2_2day = df_X_uct2_2day.rename(columns={'uct_click_today':'uct_click_2day'});
df_X_uct2_2day = df_X_uct2_2day.set_index(['user_id','item_category', 'time']);

df_X_uct3_2day = df_X_uct3_today.reset_index(level=[0, 1, 2]);
df_X_uct3_2day.time = df_X_uct3_2day.time.add(np.timedelta64(2, 'D'));
df_X_uct3_2day = df_X_uct3_2day.rename(columns={'uct_collec_today':'uct_collec_2day'});
df_X_uct3_2day = df_X_uct3_2day.set_index(['user_id', 'item_category', 'time']);

df_X_uct4_2day = df_X_uct4_today.reset_index(level=[0, 1, 2]);
df_X_uct4_2day.time = df_X_uct4_2day.time.add(np.timedelta64(2, 'D'));
df_X_uct4_2day = df_X_uct4_2day.rename(columns={'uct_cart_today':'uct_cart_2day'});
df_X_uct4_2day = df_X_uct4_2day.set_index(['user_id', 'item_category', 'time']);

#for 3-10 days before
df_X_uct_day = [];
for i in range(3, 11):
	df_X_uct_iday = df_clean.loc[:, ['user_id', 'item_category', 'time', 'behavior_type']];
	df_X_uct_iday.time = df_X_uct_iday.time.add(np.timedelta64(int(i), 'D'));
	df_X_uctn = [];
	df_X_uctn.append(df_X_uct_iday.loc[df_X_uct_iday.behavior_type==4].groupby(['user_id', 'item_category', 'time']).size().reset_index(name='uct_buy_' + str(i) + 'day'));
	df_X_uctn.append(df_X_uct_iday.loc[df_X_uct_iday.behavior_type==1].groupby(['user_id', 'item_category', 'time']).size().reset_index(name='uct_click_' + str(i) + 'day'));
	df_X_uctn.append(df_X_uct_iday.loc[df_X_uct_iday.behavior_type==2].groupby(['user_id', 'item_category', 'time']).size().reset_index(name='uct_collec_' + str(i) + 'day'));
	df_X_uctn.append(df_X_uct_iday.loc[df_X_uct_iday.behavior_type==3].groupby(['user_id', 'item_category', 'time']).size().reset_index(name='uct_cart_' + str(i) + 'day'));
	df_X_uct_day.append(df_X_uctn);
df_X_uct1_310day = df_X_uct_day[0][0].rename(columns={'uct_buy_3day':'uct_buy_310day'}).set_index(['user_id', 'item_category', 'time']);
df_X_uct2_310day = df_X_uct_day[0][1].rename(columns={'uct_click_3day':'uct_click_310day'}).set_index(['user_id', 'item_category', 'time']);
df_X_uct3_310day = df_X_uct_day[0][2].rename(columns={'uct_collec_3day':'uct_collec_310day'}).set_index(['user_id', 'item_category', 'time']);
df_X_uct4_310day = df_X_uct_day[0][3].rename(columns={'uct_cart_3day':'uct_cart_310day'}).set_index(['user_id', 'item_category', 'time']);
df_X_uct1_310day.uct_buy_310day = 0;
df_X_uct2_310day.uct_click_310day = 0;
df_X_uct3_310day.uct_collec_310day = 0;
df_X_uct4_310day.uct_cart_310day = 0;
for i in range(0, 8):
    df_X_uct1_310day = df_X_uct1_310day.join(df_X_uct_day[i][0].set_index(['user_id', 'item_category', 'time']), how='outer');
    df_X_uct2_310day = df_X_uct2_310day.join(df_X_uct_day[i][1].set_index(['user_id', 'item_category', 'time']), how='outer');
    df_X_uct3_310day = df_X_uct3_310day.join(df_X_uct_day[i][2].set_index(['user_id', 'item_category', 'time']), how='outer');
    df_X_uct4_310day = df_X_uct4_310day.join(df_X_uct_day[i][3].set_index(['user_id', 'item_category', 'time']), how='outer');
df_X_uct1_310day.fillna(0, inplace=True);
df_X_uct2_310day.fillna(0, inplace=True);
df_X_uct3_310day.fillna(0, inplace=True);
df_X_uct4_310day.fillna(0, inplace=True);
for i in range(1, 9):
    df_X_uct1_310day.ix[:, 0] = df_X_uct1_310day.ix[:, 0] + df_X_uct1_310day.ix[:, i] * (0.65 ** (np.log((i+2)**2 + 1))); #time decay
    df_X_uct2_310day.ix[:, 0] = df_X_uct2_310day.ix[:, 0] + df_X_uct2_310day.ix[:, i] * (0.65 ** (np.log((i+2)**2 + 1))); 
    df_X_uct3_310day.ix[:, 0] = df_X_uct3_310day.ix[:, 0] + df_X_uct3_310day.ix[:, i] * (0.65 ** (np.log((i+2)**2 + 1))); 
    df_X_uct4_310day.ix[:, 0] = df_X_uct4_310day.ix[:, 0] + df_X_uct4_310day.ix[:, i] * (0.65 ** (np.log((i+2)**2 + 1))); 
df_X_uct1_310day.drop(df_X_uct1_310day.columns[[1, 2, 3, 4, 5, 6, 7, 8]], axis=1, inplace=True);
df_X_uct2_310day.drop(df_X_uct2_310day.columns[[1, 2, 3, 4, 5, 6, 7, 8]], axis=1, inplace=True);
df_X_uct3_310day.drop(df_X_uct3_310day.columns[[1, 2, 3, 4, 5, 6, 7, 8]], axis=1, inplace=True);
df_X_uct4_310day.drop(df_X_uct4_310day.columns[[1, 2, 3, 4, 5, 6, 7, 8]], axis=1, inplace=True);

df_X_uct5_310day = df_X_uct1_310day.rename(columns={'uct_buy_310day':'uct_buy_prc_310day'});
df_X_uct5_310day = df_X_uct5_310day.join(df_X_uct1_310day, how='outer').reset_index(level=[1]).join(df_X_ut1_310day, how='left');
df_X_uct5_310day.uct_buy_prc_310day = df_X_uct5_310day.uct_buy_310day / df_X_uct5_310day.ut_buy_310day;
df_X_uct5_310day.set_index('item_category', append=True, inplace=True);
df_X_uct5_310day = df_X_uct5_310day.reorder_levels(['user_id', 'item_category', 'time']);
df_X_uct5_310day.drop(df_X_uct5_310day.columns[[1, 2]], axis=1, inplace=True);

df_X_uct6_310day = df_X_uct2_310day.rename(columns={'uct_click_310day':'uct_ck_prc_310day'});
df_X_uct6_310day = df_X_uct6_310day.join(df_X_uct2_310day, how='outer').reset_index(level=[1]).join(df_X_ut2_310day, how='left');
df_X_uct6_310day.uct_click_prc_310day = df_X_uct6_310day.uct_click_310day / df_X_uct6_310day.ut_click_310day;
df_X_uct6_310day.set_index('item_category', append=True, inplace=True);
df_X_uct6_310day = df_X_uct6_310day.reorder_levels(['user_id', 'item_category', 'time']);
df_X_uct6_310day.drop(df_X_uct6_310day.columns[[1, 2]], axis=1, inplace=True);

df_X_uct7_310day = df_X_uct3_310day.rename(columns={'uct_collec_310day':'uct_cl_prc_310day'});
df_X_uct7_310day = df_X_uct7_310day.join(df_X_uct3_310day, how='outer').reset_index(level=[1]).join(df_X_ut3_310day, how='left');
df_X_uct7_310day.uct_cl_prc_310day = df_X_uct7_310day.uct_collec_310day / df_X_uct7_310day.ut_collec_310day;
df_X_uct7_310day.set_index('item_category', append=True, inplace=True);
df_X_uct7_310day = df_X_uct7_310day.reorder_levels(['user_id', 'item_category', 'time']);
df_X_uct7_310day.drop(df_X_uct7_310day.columns[[1, 2]], axis=1, inplace=True);

df_X_uct8_310day = df_X_uct4_310day.rename(columns={'uct_cart_310day':'uct_ct_prc_310day'});
df_X_uct8_310day = df_X_uct8_310day.join(df_X_uct4_310day, how='outer').reset_index(level=[1]).join(df_X_ut4_310day, how='left');
df_X_uct8_310day.uct_ct_prc_310day = df_X_uct8_310day.uct_cart_310day / df_X_uct8_310day.ut_cart_310day;
df_X_uct8_310day.set_index('item_category', append=True, inplace=True);
df_X_uct8_310day = df_X_uct8_310day.reorder_levels(['user_id', 'item_category', 'time']);
df_X_uct8_310day.drop(df_X_uct8_310day.columns[[1, 2]], axis=1, inplace=True);

df_X_uct9_310day = df_X_uct1_310day.rename(columns={'uct_buy_310day':'uct_buy_prt_310day'});
df_X_uct9_310day = df_X_uct9_310day.join(df_X_uct1_310day, how='outer').reset_index(level=[2]).join(df_X_uc1, how='left');
df_X_uct9_310day.uct_buy_prt_310day = df_X_uct9_310day.uct_buy_310day / df_X_uct9_310day.uc_buy;
df_X_uct9_310day.set_index('time', append=True, inplace=True);
df_X_uct9_310day = df_X_uct9_310day.reorder_levels(['user_id', 'item_category', 'time']);
df_X_uct9_310day.drop(df_X_uct9_310day.columns[[1, 2]],axis=1, inplace=True);

df_X_uct10_310day = df_X_uct2_310day.rename(columns={'uct_click_310day':'uct_ck_prt_310day'});
df_X_uct10_310day = df_X_uct10_310day.join(df_X_uct2_310day, how='outer').reset_index(level=[2]).join(df_X_uc2, how='left');
df_X_uct10_310day.uct_ck_prt_310day = df_X_uct10_310day.uct_click_310day / df_X_uct10_310day.uc_click;
df_X_uct10_310day.set_index('time', append=True, inplace=True);
df_X_uct10_310day = df_X_uct10_310day.reorder_levels(['user_id', 'item_category', 'time']);
df_X_uct10_310day.drop(df_X_uct10_310day.columns[[1, 2]],axis=1, inplace=True);

df_X_uct11_310day = df_X_uct3_310day.rename(columns={'uct_collec_310day':'uct_cl_prt_310day'});
df_X_uct11_310day = df_X_uct11_310day.join(df_X_uct3_310day, how='outer').reset_index(level=[2]).join(df_X_uc3, how='left');
df_X_uct11_310day.uct_cl_prt_310day = df_X_uct11_310day.uct_collec_310day / df_X_uct11_310day.uc_collec;
df_X_uct11_310day.set_index('time', append=True, inplace=True);
df_X_uct11_310day = df_X_uct11_310day.reorder_levels(['user_id', 'item_category', 'time']);
df_X_uct11_310day.drop(df_X_uct11_310day.columns[[1, 2]],axis=1, inplace=True);

df_X_uct12_310day = df_X_uct4_310day.rename(columns={'uct_cart_310day':'uct_ct_prt_310day'});
df_X_uct12_310day = df_X_uct12_310day.join(df_X_uct4_310day, how='outer').reset_index(level=[2]).join(df_X_uc4, how='left');
df_X_uct12_310day.uct_ct_prt_310day = df_X_uct12_310day.uct_cart_310day / df_X_uct12_310day.uc_cart;
df_X_uct12_310day.set_index('time', append=True, inplace=True);
df_X_uct12_310day = df_X_uct12_310day.reorder_levels(['user_id', 'item_category', 'time']);
df_X_uct12_310day.drop(df_X_uct12_310day.columns[[1, 2]],axis=1, inplace=True);

#for 0-10 days before
df_X_uct_day = [];
for i in range(0, 11):
	df_X_uct_iday = df_clean.loc[:, ['user_id', 'item_category', 'time', 'behavior_type']];
	df_X_uct_iday.time = df_X_uct_iday.time.add(np.timedelta64(int(i), 'D'));
	df_X_uct_day.append(df_X_uct_iday.loc[df_X_uct_iday.behavior_type==4].groupby(['user_id', 'item_category', 'time']).size().reset_index(name='uct_buy_' + str(i) + 'day'));
df_X_uct1_010day = df_X_uct_day[0].rename(columns={'uct_buy_0day':'uct_co_buy_pre_010day'}).set_index(['user_id', 'item_category', 'time']);
df_X_uct1_010day.uct_co_buy_pre_010day = 0;
for i in range(0, 11):
	df_X_uct1_010day = df_X_uct1_010day.join(df_X_uct_day[i].set_index(['user_id', 'item_category', 'time']), how='outer');
df_X_uct1_010day.fillna(0, inplace=True);
df_X_uct1_010day = (df_X_uct1_010day!=0).astype(int);
for i in range(1, 12):
	df_X_uct1_010day.ix[:, 0] = df_X_uct1_010day.ix[:, 0] + df_X_uct1_010day.ix[:, i];
for i in range(1, 11):
	df_X_uct1_010day.ix[:, i] = (df_X_uct1_010day.ix[:, i].astype(bool) & df_X_uct1_010day.ix[:, i+1].astype(bool));
	df_X_uct1_010day.ix[:, i] = df_X_uct1_010day.ix[:, i].astype(int);
for i in range(2, 11):
	df_X_uct1_010day.ix[:, 1] = df_X_uct1_010day.ix[:, 1] + df_X_uct1_010day.ix[:, i];
df_X_uct1_010day.ix[:, 0] = df_X_uct1_010day.ix[:, 1] / df_X_uct1_010day.ix[:, 0];
df_X_uct1_010day.drop(df_X_uct1_010day.columns[[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]], axis=1, inplace=True);

#for the whole time
df_X_uct1 = df_clean.loc[:, ['user_id', 'item_category', 'time', 'behavior_type']].groupby(['user_id', 'item_category', 'time']).size().reset_index(name='uct_bh');
df_X_uct1.drop(['uct_bh'], axis=1, inplace=True);
df_X_uct1m = df_clean.loc[df_clean.behavior_type==4].groupby(['user_id', 'item_category', 'time']).size().reset_index(name='uct_buy');
df_X_uct1m.drop(['uct_buy'], axis=1, inplace=True);
df_X_uct1 = df_X_uct1.join(df_X_uct1m.set_index(['user_id', 'item_category']), on=['user_id', 'item_category'], rsuffix='_m', how='left');
df_X_uct1 = df_X_uct1.set_index(['user_id', 'item_category', 'time'], drop=False);
df_X_uct1.time_m = df_X_uct1.time - df_X_uct1.time_m;
df_X_uct1 = df_X_uct1.loc[df_X_uct1.time_m>np.timedelta64(0, 'D')];
df_X_uct1 = df_X_uct1.groupby(['user_id', 'item_category', 'time']).min();
df_X_uct1 = df_X_uct1.rename(columns={'time_m':'uct_buy_lasttime'});
df_X_uct1.uct_buy_lasttime = df_X_uct1.uct_buy_lasttime.dt.days;

df_X_uct2 = df_clean.loc[:, ['user_id', 'item_category', 'time', 'behavior_type']].groupby(['user_id', 'item_category', 'time']).size().reset_index(name='uct_bh');
df_X_uct2.drop(['uct_bh'], axis=1, inplace=True);
df_X_uct2m = df_X_uct2.loc[:, ['user_id', 'item_category', 'time']];
df_X_uct2m.index = df_X_uct2m.index + 1;
df_X_uct2 = df_X_uct2.join(df_X_uct2m, rsuffix='_m', how='left');
df_X_uct2 = df_X_uct2.set_index(['user_id', 'item_category', 'time'], drop=False);
df_X_uct2.user_id = df_X_uct2.user_id - df_X_uct2.user_id_m;
df_X_uct2.item_category = df_X_uct2.item_category - df_X_uct2.item_category_m;
df_X_uct2.time = df_X_uct2.time - df_X_uct2.time_m;
df_X_uct2 = df_X_uct2.loc[(df_X_uct2.user_id==0) & (df_X_uct2.item_category==0)];
df_X_uct2 = df_X_uct2.rename(columns={'time':'uct_bh_lasttime'});
df_X_uct2.uct_bh_lasttime = df_X_uct2.uct_bh_lasttime.dt.days;
df_X_uct2.drop(['user_id', 'user_id_m', 'item_category', 'item_category_m', 'time_m'], axis=1, inplace=True);

#for today
df_X_uit1_today = df_clean.loc[df_clean.behavior_type==4].groupby(['user_id', 'item_id', 'time']).size().reset_index(name='uit_buy_today').set_index(['user_id', 'item_id', 'time']);
df_X_uit2_today = df_clean.loc[df_clean.behavior_type==1].groupby(['user_id', 'item_id', 'time']).size().reset_index(name='uit_click_today').set_index(['user_id', 'item_id', 'time']);
df_X_uit3_today = df_clean.loc[df_clean.behavior_type==2].groupby(['user_id', 'item_id', 'time']).size().reset_index(name='uit_collec_today').set_index(['user_id', 'item_id', 'time']);
df_X_uit4_today = df_clean.loc[df_clean.behavior_type==3].groupby(['user_id', 'item_id', 'time']).size().reset_index(name='uit_cart_today').set_index(['user_id', 'item_id', 'time']);

#for one day before
df_X_uit1_1day = df_X_uit1_today.reset_index(level=[0, 1, 2]);
df_X_uit1_1day.time = df_X_uit1_1day.time.add(np.timedelta64(1, 'D'));
df_X_uit1_1day = df_X_uit1_1day.rename(columns={'uit_buy_today':'uit_buy_1day'});
df_X_uit1_1day = df_X_uit1_1day.set_index(['user_id', 'item_id', 'time']);

df_X_uit2_1day = df_X_uit2_today.reset_index(level=[0, 1, 2]);
df_X_uit2_1day.time = df_X_uit2_1day.time.add(np.timedelta64(1, 'D'));
df_X_uit2_1day = df_X_uit2_1day.rename(columns={'uit_click_today':'uit_click_1day'});
df_X_uit2_1day = df_X_uit2_1day.set_index(['user_id','item_id', 'time']);

df_X_uit3_1day = df_X_uit3_today.reset_index(level=[0, 1, 2]);
df_X_uit3_1day.time = df_X_uit3_1day.time.add(np.timedelta64(1, 'D'));
df_X_uit3_1day = df_X_uit3_1day.rename(columns={'uit_collec_today':'uit_collec_1day'});
df_X_uit3_1day = df_X_uit3_1day.set_index(['user_id', 'item_id', 'time']);

df_X_uit4_1day = df_X_uit4_today.reset_index(level=[0, 1, 2]);
df_X_uit4_1day.time = df_X_uit4_1day.time.add(np.timedelta64(1, 'D'));
df_X_uit4_1day = df_X_uit4_1day.rename(columns={'uit_cart_today':'uit_cart_1day'});
df_X_uit4_1day = df_X_uit4_1day.set_index(['user_id', 'item_id', 'time']);

#for 2 days before
df_X_uit1_2day = df_X_uit1_today.reset_index(level=[0, 1, 2]);
df_X_uit1_2day.time = df_X_uit1_2day.time.add(np.timedelta64(2, 'D'));
df_X_uit1_2day = df_X_uit1_2day.rename(columns={'uit_buy_today':'uit_buy_2day'});
df_X_uit1_2day = df_X_uit1_2day.set_index(['user_id', 'item_id', 'time']);

df_X_uit2_2day = df_X_uit2_today.reset_index(level=[0, 1, 2]);
df_X_uit2_2day.time = df_X_uit2_2day.time.add(np.timedelta64(2, 'D'));
df_X_uit2_2day = df_X_uit2_2day.rename(columns={'uit_click_today':'uit_click_2day'});
df_X_uit2_2day = df_X_uit2_2day.set_index(['user_id','item_id', 'time']);

df_X_uit3_2day = df_X_uit3_today.reset_index(level=[0, 1, 2]);
df_X_uit3_2day.time = df_X_uit3_2day.time.add(np.timedelta64(2, 'D'));
df_X_uit3_2day = df_X_uit3_2day.rename(columns={'uit_collec_today':'uit_collec_2day'});
df_X_uit3_2day = df_X_uit3_2day.set_index(['user_id', 'item_id', 'time']);

df_X_uit4_2day = df_X_uit4_today.reset_index(level=[0, 1, 2]);
df_X_uit4_2day.time = df_X_uit4_2day.time.add(np.timedelta64(2, 'D'));
df_X_uit4_2day = df_X_uit4_2day.rename(columns={'uit_cart_today':'uit_cart_2day'});
df_X_uit4_2day = df_X_uit4_2day.set_index(['user_id', 'item_id', 'time']);

#for 3-10 days before
df_X_uit_day = [];
for i in range(3, 11):
	df_X_uit_iday = df_clean.loc[:, ['user_id', 'item_id', 'time', 'behavior_type']];
	df_X_uit_iday.time = df_X_uit_iday.time.add(np.timedelta64(int(i), 'D'));
	df_X_uitn = [];
	df_X_uitn.append(df_X_uit_iday.loc[df_X_uit_iday.behavior_type==4].groupby(['user_id', 'item_id', 'time']).size().reset_index(name='uit_buy_' + str(i) + 'day'));
	df_X_uitn.append(df_X_uit_iday.loc[df_X_uit_iday.behavior_type==1].groupby(['user_id', 'item_id', 'time']).size().reset_index(name='uit_click_' + str(i) + 'day'));
	df_X_uitn.append(df_X_uit_iday.loc[df_X_uit_iday.behavior_type==2].groupby(['user_id', 'item_id', 'time']).size().reset_index(name='uit_collec_' + str(i) + 'day'));
	df_X_uitn.append(df_X_uit_iday.loc[df_X_uit_iday.behavior_type==3].groupby(['user_id', 'item_id', 'time']).size().reset_index(name='uit_cart_' + str(i) + 'day'));
	df_X_uit_day.append(df_X_uitn);
df_X_uit1_310day = df_X_uit_day[0][0].rename(columns={'uit_buy_3day':'uit_buy_310day'}).set_index(['user_id', 'item_id', 'time']);
df_X_uit2_310day = df_X_uit_day[0][1].rename(columns={'uit_click_3day':'uit_click_310day'}).set_index(['user_id', 'item_id', 'time']);
df_X_uit3_310day = df_X_uit_day[0][2].rename(columns={'uit_collec_3day':'uit_collec_310day'}).set_index(['user_id', 'item_id', 'time']);
df_X_uit4_310day = df_X_uit_day[0][3].rename(columns={'uit_cart_3day':'uit_cart_310day'}).set_index(['user_id', 'item_id', 'time']);
df_X_uit1_310day.uit_buy_310day = 0;
df_X_uit2_310day.uit_click_310day = 0;
df_X_uit3_310day.uit_collec_310day = 0;
df_X_uit4_310day.uit_cart_310day = 0;
for i in range(0, 8):
    df_X_uit1_310day = df_X_uit1_310day.join(df_X_uit_day[i][0].set_index(['user_id', 'item_id', 'time']), how='outer');
    df_X_uit2_310day = df_X_uit2_310day.join(df_X_uit_day[i][1].set_index(['user_id', 'item_id', 'time']), how='outer');
    df_X_uit3_310day = df_X_uit3_310day.join(df_X_uit_day[i][2].set_index(['user_id', 'item_id', 'time']), how='outer');
    df_X_uit4_310day = df_X_uit4_310day.join(df_X_uit_day[i][3].set_index(['user_id', 'item_id', 'time']), how='outer');
df_X_uit1_310day.fillna(0, inplace=True);
df_X_uit2_310day.fillna(0, inplace=True);
df_X_uit3_310day.fillna(0, inplace=True);
df_X_uit4_310day.fillna(0, inplace=True);
for i in range(1, 9):
    df_X_uit1_310day.ix[:, 0] = df_X_uit1_310day.ix[:, 0] + df_X_uit1_310day.ix[:, i] * (0.65 ** (np.log((i+2)**2 + 1))); #time decay
    df_X_uit2_310day.ix[:, 0] = df_X_uit2_310day.ix[:, 0] + df_X_uit2_310day.ix[:, i] * (0.65 ** (np.log((i+2)**2 + 1))); 
    df_X_uit3_310day.ix[:, 0] = df_X_uit3_310day.ix[:, 0] + df_X_uit3_310day.ix[:, i] * (0.65 ** (np.log((i+2)**2 + 1))); 
    df_X_uit4_310day.ix[:, 0] = df_X_uit4_310day.ix[:, 0] + df_X_uit4_310day.ix[:, i] * (0.65 ** (np.log((i+2)**2 + 1))); 
df_X_uit1_310day.drop(df_X_uit1_310day.columns[[1, 2, 3, 4, 5, 6, 7, 8]], axis=1, inplace=True);
df_X_uit2_310day.drop(df_X_uit2_310day.columns[[1, 2, 3, 4, 5, 6, 7, 8]], axis=1, inplace=True);
df_X_uit3_310day.drop(df_X_uit3_310day.columns[[1, 2, 3, 4, 5, 6, 7, 8]], axis=1, inplace=True);
df_X_uit4_310day.drop(df_X_uit4_310day.columns[[1, 2, 3, 4, 5, 6, 7, 8]], axis=1, inplace=True);

df_X_uit5_310day = df_X_uit1_310day.rename(columns={'uit_buy_310day':'uit_buy_prc_310day'});
df_X_uit5_310day = df_X_uit5_310day.join(df_X_uit1_310day, how='outer').join(df2.set_index('item_id'), how='left').reset_index(level=[1]).set_index('item_category', append=True).reorder_levels(['user_id', 'item_category', 'time']).join(df_X_uct1_310day, how='left');
df_X_uit5_310day.uit_buy_prc_310day = df_X_uit5_310day.uit_buy_310day / df_X_uit5_310day.uct_buy_310day;
df_X_uit5_310day = df_X_uit5_310day.reset_index(level=[1]).set_index('item_id', append=True).reorder_levels(['user_id', 'item_id', 'time']);
df_X_uit5_310day.drop(df_X_uit5_310day.columns[[0, 2, 3]], axis=1, inplace=True);

df_X_uit6_310day = df_X_uit2_310day.rename(columns={'uit_click_310day':'uit_ck_prc_310day'});
df_X_uit6_310day = df_X_uit6_310day.join(df_X_uit2_310day, how='outer').join(df2.set_index('item_id'), how='left').reset_index(level=[1]).set_index('item_category', append=True).reorder_levels(['user_id', 'item_category', 'time']).join(df_X_uct2_310day, how='left');
df_X_uit6_310day.uit_click_prc_310day = df_X_uit6_310day.uit_click_310day / df_X_uit6_310day.uct_click_310day;
df_X_uit6_310day = df_X_uit6_310day.reset_index(level=[1]).set_index('item_id', append=True).reorder_levels(['user_id', 'item_id', 'time']);
df_X_uit6_310day.drop(df_X_uit6_310day.columns[[0, 2, 3]], axis=1, inplace=True);

df_X_uit7_310day = df_X_uit3_310day.rename(columns={'uit_collec_310day':'uit_cl_prc_310day'});
df_X_uit7_310day = df_X_uit7_310day.join(df_X_uit3_310day, how='outer').join(df2.set_index('item_id'), how='left').reset_index(level=[1]).set_index('item_category', append=True).reorder_levels(['user_id', 'item_category', 'time']).join(df_X_uct3_310day, how='left');
df_X_uit7_310day.uit_cl_prc_310day = df_X_uit7_310day.uit_collec_310day / df_X_uit7_310day.uct_collec_310day;
df_X_uit7_310day = df_X_uit7_310day.reset_index(level=[1]).set_index('item_id', append=True).reorder_levels(['user_id', 'item_id', 'time']);
df_X_uit7_310day.drop(df_X_uit7_310day.columns[[0, 2, 3]], axis=1, inplace=True);

df_X_uit8_310day = df_X_uit4_310day.rename(columns={'uit_cart_310day':'uit_ct_prc_310day'});
df_X_uit8_310day = df_X_uit8_310day.join(df_X_uit4_310day, how='outer').join(df2.set_index('item_id'), how='left').reset_index(level=[1]).set_index('item_category', append=True).reorder_levels(['user_id', 'item_category', 'time']).join(df_X_uct4_310day, how='left');
df_X_uit8_310day.uit_ct_prc_310day = df_X_uit8_310day.uit_cart_310day / df_X_uit8_310day.uct_cart_310day;
df_X_uit8_310day = df_X_uit8_310day.reset_index(level=[1]).set_index('item_id', append=True).reorder_levels(['user_id', 'item_id', 'time']);
df_X_uit8_310day.drop(df_X_uit8_310day.columns[[0, 2, 3]], axis=1, inplace=True);

#for 0-10 days before
df_X_uit_day = [];
for i in range(0, 11):
	df_X_uit_iday = df_clean.loc[:, ['user_id', 'item_id', 'time', 'behavior_type']];
	df_X_uit_iday.time = df_X_uit_iday.time.add(np.timedelta64(int(i), 'D'));
	df_X_uit_day.append(df_X_uit_iday.loc[df_X_uit_iday.behavior_type==4].groupby(['user_id', 'item_id', 'time']).size().reset_index(name='uit_buy_' + str(i) + 'day'));
df_X_uit1_010day = df_X_uit_day[0].rename(columns={'uit_buy_0day':'uit_co_buy_pre_010day'}).set_index(['user_id', 'item_id', 'time']);
df_X_uit1_010day.uit_co_buy_pre_010day = 0;
for i in range(0, 11):
	df_X_uit1_010day = df_X_uit1_010day.join(df_X_uit_day[i].set_index(['user_id', 'item_id', 'time']), how='outer');
df_X_uit1_010day.fillna(0, inplace=True);
df_X_uit1_010day = (df_X_uit1_010day!=0).astype(int);
for i in range(1, 12):
	df_X_uit1_010day.ix[:, 0] = df_X_uit1_010day.ix[:, 0] + df_X_uit1_010day.ix[:, i];
for i in range(1, 11):
	df_X_uit1_010day.ix[:, i] = (df_X_uit1_010day.ix[:, i].astype(bool) & df_X_uit1_010day.ix[:, i+1].astype(bool));
	df_X_uit1_010day.ix[:, i] = df_X_uit1_010day.ix[:, i].astype(int);
for i in range(2, 11):
	df_X_uit1_010day.ix[:, 1] = df_X_uit1_010day.ix[:, 1] + df_X_uit1_010day.ix[:, i];
df_X_uit1_010day.ix[:, 0] = df_X_uit1_010day.ix[:, 1] / df_X_uit1_010day.ix[:, 0];
df_X_uit1_010day.drop(df_X_uit1_010day.columns[[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]], axis=1, inplace=True);

#for the while time
df_X_uit1 = df_clean.loc[:, ['user_id', 'item_id', 'time', 'behavior_type']].groupby(['user_id', 'item_id', 'time']).size().reset_index(name='uit_bh');
df_X_uit1.drop(['uit_bh'], axis=1, inplace=True);
df_X_uit1m = df_clean.loc[df_clean.behavior_type==4].groupby(['user_id', 'item_id', 'time']).size().reset_index(name='uit_buy');
df_X_uit1m.drop(['uit_buy'], axis=1, inplace=True);
df_X_uit1 = df_X_uit1.join(df_X_uit1m.set_index(['user_id', 'item_id']), on=['user_id', 'item_id'], rsuffix='_m', how='left');
df_X_uit1 = df_X_uit1.set_index(['user_id', 'item_id', 'time'], drop=False);
df_X_uit1.time_m = df_X_uit1.time - df_X_uit1.time_m;
df_X_uit1 = df_X_uit1.loc[df_X_uit1.time_m>np.timedelta64(0, 'D')];
df_X_uit1 = df_X_uit1.groupby(['user_id', 'item_id', 'time']).min();
df_X_uit1 = df_X_uit1.rename(columns={'time_m':'uit_buy_lasttime'});
df_X_uit1.uit_buy_lasttime = df_X_uit1.uit_buy_lasttime.dt.days;

df_X_uit2 = df_clean.loc[:, ['user_id', 'item_id', 'time', 'behavior_type']].groupby(['user_id', 'item_id', 'time']).size().reset_index(name='uit_bh');
df_X_uit2.drop(['uit_bh'], axis=1, inplace=True);
df_X_uit2m = df_X_uit2.loc[:, ['user_id', 'item_id', 'time']];
df_X_uit2m.index = df_X_uit2m.index + 1;
df_X_uit2 = df_X_uit2.join(df_X_uit2m, rsuffix='_m', how='left');
df_X_uit2 = df_X_uit2.set_index(['user_id', 'item_id', 'time'], drop=False);
df_X_uit2.user_id = df_X_uit2.user_id - df_X_uit2.user_id_m;
df_X_uit2.item_id = df_X_uit2.item_id - df_X_uit2.item_id_m;
df_X_uit2.time = df_X_uit2.time - df_X_uit2.time_m;
df_X_uit2 = df_X_uit2.loc[(df_X_uit2.user_id==0) & (df_X_uit2.item_id==0)];
df_X_uit2 = df_X_uit2.rename(columns={'time':'uit_bh_lasttime'});
df_X_uit2.uit_bh_lasttime = df_X_uit2.uit_bh_lasttime.dt.days;
df_X_uit2.drop(['user_id', 'user_id_m', 'item_id', 'item_id_m', 'time_m'], axis=1, inplace=True);

##build X_Featrue
df_X = df_clean.loc[:, ['user_id', 'item_id', 'item_category', 'time']];
df_X.drop_duplicates(inplace=True);

#join user
df_X = df_X.join(df_X_u1, on='user_id', how='left').fillna(0);
df_X = df_X.join(df_X_u2, on='user_id', how='left').fillna(0);
df_X = df_X.join(df_X_u3, on='user_id', how='left').fillna(0);
df_X = df_X.join(df_X_u4, on='user_id', how='left').fillna(0);
df_X = df_X.join(df_X_u5, on='user_id', how='left').fillna(0);
df_X = df_X.join(df_X_u6, on='user_id', how='left').fillna(0);
df_X = df_X.join(df_X_u7, on='user_id', how='left').fillna(0);
df_X = df_X.join(df_X_u8, on='user_id', how='left').fillna(0);
df_X = df_X.join(df_X_u9, on='user_id', how='left').fillna(0);
df_X = df_X.join(df_X_u10, on='user_id', how='left').fillna(0);

#join item(category)
df_X = df_X.join(df_X_c1, on='item_category', how='left').fillna(0);
df_X = df_X.join(df_X_c2, on='item_category', how='left').fillna(0);
df_X = df_X.join(df_X_c3, on='item_category', how='left').fillna(0);
df_X = df_X.join(df_X_c4, on='item_category', how='left').fillna(0);
df_X = df_X.join(df_X_c5, on='item_category', how='left').fillna(0);
df_X = df_X.join(df_X_c6, on='item_category', how='left').fillna(0);
df_X = df_X.join(df_X_c7, on='item_category', how='left').fillna(0);
df_X = df_X.join(df_X_c8, on='item_category', how='left').fillna(0);
df_X = df_X.join(df_X_c9, on='item_category', how='left').fillna(0);
df_X = df_X.join(df_X_c10, on='item_category', how='left').fillna(0);
df_X = df_X.join(df_X_c11, on='item_category', how='left').fillna(0);
df_X = df_X.join(df_X_c12, on='item_category', how='left').fillna(0);
df_X = df_X.join(df_X_c13, on='item_category', how='left').fillna(0);
df_X = df_X.join(df_X_c14, on='item_category', how='left').fillna(0);

df_X = df_X.join(df_X_i1, on='item_id', how='left').fillna(0);
df_X = df_X.join(df_X_i2, on='item_id', how='left').fillna(0);
df_X = df_X.join(df_X_i3, on='item_id', how='left').fillna(0);
df_X = df_X.join(df_X_i4, on='item_id', how='left').fillna(0);
df_X = df_X.join(df_X_i5, on='item_id', how='left').fillna(0);
df_X = df_X.join(df_X_i6, on='item_id', how='left').fillna(0);
df_X = df_X.join(df_X_i7, on='item_id', how='left').fillna(0);
df_X = df_X.join(df_X_i8, on='item_id', how='left').fillna(0);
df_X = df_X.join(df_X_i9, on='item_id', how='left').fillna(0);
df_X = df_X.join(df_X_i10, on='item_id', how='left').fillna(0);
df_X = df_X.join(df_X_i11, on='item_id', how='left').fillna(0);
df_X = df_X.join(df_X_i12, on='item_id', how='left').fillna(0);
df_X = df_X.join(df_X_i13, on='item_id', how='left').fillna(0);
df_X = df_X.join(df_X_i14, on='item_id', how='left').fillna(0);

#join time
df_X = df_X.join(df_X_t1, on='time', how='left');

#join user X item
df_X = df_X.join(df_X_uc1, on=['user_id', 'item_category'], how='left').fillna(0);
df_X = df_X.join(df_X_uc2, on=['user_id', 'item_category'], how='left').fillna(0);
df_X = df_X.join(df_X_uc3, on=['user_id', 'item_category'], how='left').fillna(0);
df_X = df_X.join(df_X_uc4, on=['user_id', 'item_category'], how='left').fillna(0);
df_X = df_X.join(df_X_uc5, on=['user_id', 'item_category'], how='left').fillna(0);
df_X = df_X.join(df_X_uc6, on=['user_id', 'item_category'], how='left').fillna(0);
df_X = df_X.join(df_X_uc7, on=['user_id', 'item_category'], how='left').fillna(0);
df_X = df_X.join(df_X_uc8, on=['user_id', 'item_category'], how='left').fillna(0);
df_X = df_X.join(df_X_uc9, on=['user_id', 'item_category'], how='left').fillna(0);
df_X = df_X.join(df_X_uc10, on=['user_id', 'item_category'], how='left').fillna(0);
df_X = df_X.join(df_X_uc11, on=['user_id', 'item_category'], how='left').fillna(0);
df_X = df_X.join(df_X_uc12, on=['user_id', 'item_category'], how='left').fillna(0);
df_X = df_X.join(df_X_uc13, on=['user_id', 'item_category'], how='left').fillna(0);

df_X = df_X.join(df_X_ui1, on=['user_id', 'item_id'], how='left').fillna(0);
df_X = df_X.join(df_X_ui2, on=['user_id', 'item_id'], how='left').fillna(0);
df_X = df_X.join(df_X_ui3, on=['user_id', 'item_id'], how='left').fillna(0);
df_X = df_X.join(df_X_ui4, on=['user_id', 'item_id'], how='left').fillna(0);
df_X = df_X.join(df_X_ui5, on=['user_id', 'item_id'], how='left').fillna(0);
df_X = df_X.join(df_X_ui6, on=['user_id', 'item_id'], how='left').fillna(0);
df_X = df_X.join(df_X_ui7, on=['user_id', 'item_id'], how='left').fillna(0);
df_X = df_X.join(df_X_ui8, on=['user_id', 'item_id'], how='left').fillna(0);
df_X = df_X.join(df_X_ui9, on=['user_id', 'item_id'], how='left').fillna(0);
df_X = df_X.join(df_X_ui10, on=['user_id', 'item_id'], how='left').fillna(0);
df_X = df_X.join(df_X_ui11, on=['user_id', 'item_id'], how='left').fillna(0);
df_X = df_X.join(df_X_ui12, on=['user_id', 'item_id'], how='left').fillna(0);
df_X = df_X.join(df_X_ui13, on=['user_id', 'item_id'], how='left').fillna(0);

#join user X time
#today
df_X = df_X.join(df_X_ut1_today, on=['user_id', 'time'], how='left').fillna(0);
df_X = df_X.join(df_X_ut2_today, on=['user_id', 'time'], how='left').fillna(0);
df_X = df_X.join(df_X_ut3_today, on=['user_id', 'time'], how='left').fillna(0);
df_X = df_X.join(df_X_ut4_today, on=['user_id', 'time'], how='left').fillna(0);
df_X = df_X.join(df_X_ut5_today, on=['user_id', 'time'], how='left').fillna(0);
df_X = df_X.join(df_X_ut6_today, on=['user_id', 'time'], how='left').fillna(0);
df_X = df_X.join(df_X_ut7_today, on=['user_id', 'time'], how='left').fillna(0);
df_X = df_X.join(df_X_ut8_today, on=['user_id', 'time'], how='left').fillna(0);
df_X = df_X.join(df_X_ut9_today, on=['user_id', 'time'], how='left').fillna(0);
df_X = df_X.join(df_X_ut10_today, on=['user_id', 'time'], how='left').fillna(0);
df_X = df_X.join(df_X_ut11_today, on=['user_id', 'time'], how='left').fillna(0);
df_X = df_X.join(df_X_ut12_today, on=['user_id', 'time'], how='left').fillna(0);

#1 day
df_X = df_X.join(df_X_ut1_1day, on=['user_id', 'time'], how='left').fillna(0);
df_X = df_X.join(df_X_ut2_1day, on=['user_id', 'time'], how='left').fillna(0);
df_X = df_X.join(df_X_ut3_1day, on=['user_id', 'time'], how='left').fillna(0);
df_X = df_X.join(df_X_ut4_1day, on=['user_id', 'time'], how='left').fillna(0);
df_X = df_X.join(df_X_ut5_1day, on=['user_id', 'time'], how='left').fillna(0);
df_X = df_X.join(df_X_ut6_1day, on=['user_id', 'time'], how='left').fillna(0);
df_X = df_X.join(df_X_ut7_1day, on=['user_id', 'time'], how='left').fillna(0);
df_X = df_X.join(df_X_ut8_1day, on=['user_id', 'time'], how='left').fillna(0);
df_X = df_X.join(df_X_ut9_1day, on=['user_id', 'time'], how='left').fillna(0);
df_X = df_X.join(df_X_ut10_1day, on=['user_id', 'time'], how='left').fillna(0);
df_X = df_X.join(df_X_ut11_1day, on=['user_id', 'time'], how='left').fillna(0);
df_X = df_X.join(df_X_ut12_1day, on=['user_id', 'time'], how='left').fillna(0);

#2 day
df_X = df_X.join(df_X_ut1_2day, on=['user_id', 'time'], how='left').fillna(0);
df_X = df_X.join(df_X_ut2_2day, on=['user_id', 'time'], how='left').fillna(0);
df_X = df_X.join(df_X_ut3_2day, on=['user_id', 'time'], how='left').fillna(0);
df_X = df_X.join(df_X_ut4_2day, on=['user_id', 'time'], how='left').fillna(0);
df_X = df_X.join(df_X_ut5_2day, on=['user_id', 'time'], how='left').fillna(0);
df_X = df_X.join(df_X_ut6_2day, on=['user_id', 'time'], how='left').fillna(0);
df_X = df_X.join(df_X_ut7_2day, on=['user_id', 'time'], how='left').fillna(0);
df_X = df_X.join(df_X_ut8_2day, on=['user_id', 'time'], how='left').fillna(0);
df_X = df_X.join(df_X_ut9_2day, on=['user_id', 'time'], how='left').fillna(0);
df_X = df_X.join(df_X_ut10_2day, on=['user_id', 'time'], how='left').fillna(0);
df_X = df_X.join(df_X_ut11_2day, on=['user_id', 'time'], how='left').fillna(0);
df_X = df_X.join(df_X_ut12_2day, on=['user_id', 'time'], how='left').fillna(0);

#3-10 day
df_X = df_X.join(df_X_ut1_310day, on=['user_id', 'time'], how='left').fillna(0);
df_X = df_X.join(df_X_ut2_310day, on=['user_id', 'time'], how='left').fillna(0);
df_X = df_X.join(df_X_ut3_310day, on=['user_id', 'time'], how='left').fillna(0);
df_X = df_X.join(df_X_ut4_310day, on=['user_id', 'time'], how='left').fillna(0);
df_X = df_X.join(df_X_ut5_310day, on=['user_id', 'time'], how='left').fillna(0);
df_X = df_X.join(df_X_ut6_310day, on=['user_id', 'time'], how='left').fillna(0);
df_X = df_X.join(df_X_ut7_310day, on=['user_id', 'time'], how='left').fillna(0);
df_X = df_X.join(df_X_ut8_310day, on=['user_id', 'time'], how='left').fillna(0);
df_X = df_X.join(df_X_ut9_310day, on=['user_id', 'time'], how='left').fillna(0);
df_X = df_X.join(df_X_ut10_310day, on=['user_id', 'time'], how='left').fillna(0);
df_X = df_X.join(df_X_ut11_310day, on=['user_id', 'time'], how='left').fillna(0);
df_X = df_X.join(df_X_ut12_310day, on=['user_id', 'time'], how='left').fillna(0);

#0-10 day
df_X = df_X.join(df_X_ut1_010day, on=['user_id', 'time'], how='left').fillna(0);
df_X = df_X.join(df_X_ut2_010day, on=['user_id', 'time'], how='left').fillna(0);

#whole period
df_X = df_X.join(df_X_ut1, on=['user_id', 'time'], how='left').fillna(30);
df_X = df_X.join(df_X_ut2, on=['user_id', 'time'], how='left').fillna(30);

#join item(category) X time
#today
df_X = df_X.join(df_X_ct1_today, on=['item_category', 'time'], how='left').fillna(0);
df_X = df_X.join(df_X_ct2_today, on=['item_category', 'time'], how='left').fillna(0);
df_X = df_X.join(df_X_ct3_today, on=['item_category', 'time'], how='left').fillna(0);
df_X = df_X.join(df_X_ct4_today, on=['item_category', 'time'], how='left').fillna(0);
df_X = df_X.join(df_X_ct5_today, on=['item_category', 'time'], how='left').fillna(0);
df_X = df_X.join(df_X_ct6_today, on=['item_category', 'time'], how='left').fillna(0);
df_X = df_X.join(df_X_ct7_today, on=['item_category', 'time'], how='left').fillna(0);
df_X = df_X.join(df_X_ct8_today, on=['item_category', 'time'], how='left').fillna(0);
df_X = df_X.join(df_X_ct9_today, on=['item_category', 'time'], how='left').fillna(0);
df_X = df_X.join(df_X_ct10_today, on=['item_category', 'time'], how='left').fillna(0);
df_X = df_X.join(df_X_ct11_today, on=['item_category', 'time'], how='left').fillna(0);
df_X = df_X.join(df_X_ct12_today, on=['item_category', 'time'], how='left').fillna(0);

#1 day
df_X = df_X.join(df_X_ct1_1day, on=['item_category', 'time'], how='left').fillna(0);
df_X = df_X.join(df_X_ct2_1day, on=['item_category', 'time'], how='left').fillna(0);
df_X = df_X.join(df_X_ct3_1day, on=['item_category', 'time'], how='left').fillna(0);
df_X = df_X.join(df_X_ct4_1day, on=['item_category', 'time'], how='left').fillna(0);
df_X = df_X.join(df_X_ct5_1day, on=['item_category', 'time'], how='left').fillna(0);
df_X = df_X.join(df_X_ct6_1day, on=['item_category', 'time'], how='left').fillna(0);
df_X = df_X.join(df_X_ct7_1day, on=['item_category', 'time'], how='left').fillna(0);
df_X = df_X.join(df_X_ct8_1day, on=['item_category', 'time'], how='left').fillna(0);
df_X = df_X.join(df_X_ct9_1day, on=['item_category', 'time'], how='left').fillna(0);
df_X = df_X.join(df_X_ct10_1day, on=['item_category', 'time'], how='left').fillna(0);
df_X = df_X.join(df_X_ct11_1day, on=['item_category', 'time'], how='left').fillna(0);
df_X = df_X.join(df_X_ct12_1day, on=['item_category', 'time'], how='left').fillna(0);

#2 day
df_X = df_X.join(df_X_ct1_2day, on=['item_category', 'time'], how='left').fillna(0);
df_X = df_X.join(df_X_ct2_2day, on=['item_category', 'time'], how='left').fillna(0);
df_X = df_X.join(df_X_ct3_2day, on=['item_category', 'time'], how='left').fillna(0);
df_X = df_X.join(df_X_ct4_2day, on=['item_category', 'time'], how='left').fillna(0);
df_X = df_X.join(df_X_ct5_2day, on=['item_category', 'time'], how='left').fillna(0);
df_X = df_X.join(df_X_ct6_2day, on=['item_category', 'time'], how='left').fillna(0);
df_X = df_X.join(df_X_ct7_2day, on=['item_category', 'time'], how='left').fillna(0);
df_X = df_X.join(df_X_ct8_2day, on=['item_category', 'time'], how='left').fillna(0);
df_X = df_X.join(df_X_ct9_2day, on=['item_category', 'time'], how='left').fillna(0);
df_X = df_X.join(df_X_ct10_2day, on=['item_category', 'time'], how='left').fillna(0);
df_X = df_X.join(df_X_ct11_2day, on=['item_category', 'time'], how='left').fillna(0);
df_X = df_X.join(df_X_ct12_2day, on=['item_category', 'time'], how='left').fillna(0);

#3-10 day
df_X = df_X.join(df_X_ct1_310day, on=['item_category', 'time'], how='left').fillna(0);
df_X = df_X.join(df_X_ct2_310day, on=['item_category', 'time'], how='left').fillna(0);
df_X = df_X.join(df_X_ct3_310day, on=['item_category', 'time'], how='left').fillna(0);
df_X = df_X.join(df_X_ct4_310day, on=['item_category', 'time'], how='left').fillna(0);
df_X = df_X.join(df_X_ct5_310day, on=['item_category', 'time'], how='left').fillna(0);
df_X = df_X.join(df_X_ct6_310day, on=['item_category', 'time'], how='left').fillna(0);
df_X = df_X.join(df_X_ct7_310day, on=['item_category', 'time'], how='left').fillna(0);
df_X = df_X.join(df_X_ct8_310day, on=['item_category', 'time'], how='left').fillna(0);
df_X = df_X.join(df_X_ct9_310day, on=['item_category', 'time'], how='left').fillna(0);
df_X = df_X.join(df_X_ct10_310day, on=['item_category', 'time'], how='left').fillna(0);
df_X = df_X.join(df_X_ct11_310day, on=['item_category', 'time'], how='left').fillna(0);
df_X = df_X.join(df_X_ct12_310day, on=['item_category', 'time'], how='left').fillna(0);

#0-10 day
df_X = df_X.join(df_X_ct1_010day, on=['item_category', 'time'], how='left').fillna(0);
df_X = df_X.join(df_X_ct2_010day, on=['item_category', 'time'], how='left').fillna(0);

#whole period
df_X = df_X.join(df_X_ct1, on=['item_category', 'time'], how='left').fillna(30);
df_X = df_X.join(df_X_ct2, on=['item_category', 'time'], how='left').fillna(30);

#today
df_X = df_X.join(df_X_it1_today, on=['item_id', 'time'], how='left').fillna(0);
df_X = df_X.join(df_X_it2_today, on=['item_id', 'time'], how='left').fillna(0);
df_X = df_X.join(df_X_it3_today, on=['item_id', 'time'], how='left').fillna(0);
df_X = df_X.join(df_X_it4_today, on=['item_id', 'time'], how='left').fillna(0);
df_X = df_X.join(df_X_it5_today, on=['item_id', 'time'], how='left').fillna(0);
df_X = df_X.join(df_X_it6_today, on=['item_id', 'time'], how='left').fillna(0);
df_X = df_X.join(df_X_it7_today, on=['item_id', 'time'], how='left').fillna(0);
df_X = df_X.join(df_X_it8_today, on=['item_id', 'time'], how='left').fillna(0);
df_X = df_X.join(df_X_it9_today, on=['item_id', 'time'], how='left').fillna(0);
df_X = df_X.join(df_X_it10_today, on=['item_id', 'time'], how='left').fillna(0);
df_X = df_X.join(df_X_it11_today, on=['item_id', 'time'], how='left').fillna(0);
df_X = df_X.join(df_X_it12_today, on=['item_id', 'time'], how='left').fillna(0);

#1 day
df_X = df_X.join(df_X_it1_1day, on=['item_id', 'time'], how='left').fillna(0);
df_X = df_X.join(df_X_it2_1day, on=['item_id', 'time'], how='left').fillna(0);
df_X = df_X.join(df_X_it3_1day, on=['item_id', 'time'], how='left').fillna(0);
df_X = df_X.join(df_X_it4_1day, on=['item_id', 'time'], how='left').fillna(0);
df_X = df_X.join(df_X_it5_1day, on=['item_id', 'time'], how='left').fillna(0);
df_X = df_X.join(df_X_it6_1day, on=['item_id', 'time'], how='left').fillna(0);
df_X = df_X.join(df_X_it7_1day, on=['item_id', 'time'], how='left').fillna(0);
df_X = df_X.join(df_X_it8_1day, on=['item_id', 'time'], how='left').fillna(0);
df_X = df_X.join(df_X_it9_1day, on=['item_id', 'time'], how='left').fillna(0);
df_X = df_X.join(df_X_it10_1day, on=['item_id', 'time'], how='left').fillna(0);
df_X = df_X.join(df_X_it11_1day, on=['item_id', 'time'], how='left').fillna(0);
df_X = df_X.join(df_X_it12_1day, on=['item_id', 'time'], how='left').fillna(0);

#2 day
df_X = df_X.join(df_X_it1_2day, on=['item_id', 'time'], how='left').fillna(0);
df_X = df_X.join(df_X_it2_2day, on=['item_id', 'time'], how='left').fillna(0);
df_X = df_X.join(df_X_it3_2day, on=['item_id', 'time'], how='left').fillna(0);
df_X = df_X.join(df_X_it4_2day, on=['item_id', 'time'], how='left').fillna(0);
df_X = df_X.join(df_X_it5_2day, on=['item_id', 'time'], how='left').fillna(0);
df_X = df_X.join(df_X_it6_2day, on=['item_id', 'time'], how='left').fillna(0);
df_X = df_X.join(df_X_it7_2day, on=['item_id', 'time'], how='left').fillna(0);
df_X = df_X.join(df_X_it8_2day, on=['item_id', 'time'], how='left').fillna(0);
df_X = df_X.join(df_X_it9_2day, on=['item_id', 'time'], how='left').fillna(0);
df_X = df_X.join(df_X_it10_2day, on=['item_id', 'time'], how='left').fillna(0);
df_X = df_X.join(df_X_it11_2day, on=['item_id', 'time'], how='left').fillna(0);
df_X = df_X.join(df_X_it12_2day, on=['item_id', 'time'], how='left').fillna(0);

#3-10 day
df_X = df_X.join(df_X_it1_310day, on=['item_id', 'time'], how='left').fillna(0);
df_X = df_X.join(df_X_it2_310day, on=['item_id', 'time'], how='left').fillna(0);
df_X = df_X.join(df_X_it3_310day, on=['item_id', 'time'], how='left').fillna(0);
df_X = df_X.join(df_X_it4_310day, on=['item_id', 'time'], how='left').fillna(0);
df_X = df_X.join(df_X_it5_310day, on=['item_id', 'time'], how='left').fillna(0);
df_X = df_X.join(df_X_it6_310day, on=['item_id', 'time'], how='left').fillna(0);
df_X = df_X.join(df_X_it7_310day, on=['item_id', 'time'], how='left').fillna(0);
df_X = df_X.join(df_X_it8_310day, on=['item_id', 'time'], how='left').fillna(0);
df_X = df_X.join(df_X_it9_310day, on=['item_id', 'time'], how='left').fillna(0);
df_X = df_X.join(df_X_it10_310day, on=['item_id', 'time'], how='left').fillna(0);
df_X = df_X.join(df_X_it11_310day, on=['item_id', 'time'], how='left').fillna(0);
df_X = df_X.join(df_X_it12_310day, on=['item_id', 'time'], how='left').fillna(0);

#0-10 day
df_X = df_X.join(df_X_it1_010day, on=['item_id', 'time'], how='left').fillna(0);
df_X = df_X.join(df_X_it2_010day, on=['item_id', 'time'], how='left').fillna(0);

#whole period
df_X = df_X.join(df_X_it1, on=['item_id', 'time'], how='left').fillna(30);
df_X = df_X.join(df_X_it2, on=['item_id', 'time'], how='left').fillna(30);

#join user X item(category) X time
#today
df_X = df_X.join(df_X_uct1_today, on=['user_id', 'item_category', 'time'], how='left').fillna(0);
df_X = df_X.join(df_X_uct2_today, on=['user_id', 'item_category', 'time'], how='left').fillna(0);
df_X = df_X.join(df_X_uct3_today, on=['user_id', 'item_category', 'time'], how='left').fillna(0);
df_X = df_X.join(df_X_uct4_today, on=['user_id', 'item_category', 'time'], how='left').fillna(0);

#1 day
df_X = df_X.join(df_X_uct1_1day, on=['user_id', 'item_category', 'time'], how='left').fillna(0);
df_X = df_X.join(df_X_uct2_1day, on=['user_id', 'item_category', 'time'], how='left').fillna(0);
df_X = df_X.join(df_X_uct3_1day, on=['user_id', 'item_category', 'time'], how='left').fillna(0);
df_X = df_X.join(df_X_uct4_1day, on=['user_id', 'item_category', 'time'], how='left').fillna(0);

#2 day
df_X = df_X.join(df_X_uct1_2day, on=['user_id', 'item_category', 'time'], how='left').fillna(0);
df_X = df_X.join(df_X_uct2_2day, on=['user_id', 'item_category', 'time'], how='left').fillna(0);
df_X = df_X.join(df_X_uct3_2day, on=['user_id', 'item_category', 'time'], how='left').fillna(0);
df_X = df_X.join(df_X_uct4_2day, on=['user_id', 'item_category', 'time'], how='left').fillna(0);

#3-10 day
df_X = df_X.join(df_X_uct1_310day, on=['user_id', 'item_category', 'time'], how='left').fillna(0);
df_X = df_X.join(df_X_uct2_310day, on=['user_id', 'item_category', 'time'], how='left').fillna(0);
df_X = df_X.join(df_X_uct3_310day, on=['user_id', 'item_category', 'time'], how='left').fillna(0);
df_X = df_X.join(df_X_uct4_310day, on=['user_id', 'item_category', 'time'], how='left').fillna(0);
df_X = df_X.join(df_X_uct5_310day, on=['user_id', 'item_category', 'time'], how='left').fillna(0);
df_X = df_X.join(df_X_uct6_310day, on=['user_id', 'item_category', 'time'], how='left').fillna(0);
df_X = df_X.join(df_X_uct7_310day, on=['user_id', 'item_category', 'time'], how='left').fillna(0);
df_X = df_X.join(df_X_uct8_310day, on=['user_id', 'item_category', 'time'], how='left').fillna(0);
df_X = df_X.join(df_X_uct9_310day, on=['user_id', 'item_category', 'time'], how='left').fillna(0);
df_X = df_X.join(df_X_uct10_310day, on=['user_id', 'item_category', 'time'], how='left').fillna(0);
df_X = df_X.join(df_X_uct11_310day, on=['user_id', 'item_category', 'time'], how='left').fillna(0);
df_X = df_X.join(df_X_uct12_310day, on=['user_id', 'item_category', 'time'], how='left').fillna(0);

#0-10 day
df_X = df_X.join(df_X_uct1_010day, on=['user_id', 'item_category', 'time'], how='left').fillna(0);

#whole period
df_X = df_X.join(df_X_uct1, on=['user_id', 'item_category', 'time'], how='left').fillna(30);
df_X = df_X.join(df_X_uct2, on=['user_id', 'item_category', 'time'], how='left').fillna(30);

#today
df_X = df_X.join(df_X_uit1_today, on=['user_id', 'item_id', 'time'], how='left').fillna(0);
df_X = df_X.join(df_X_uit2_today, on=['user_id', 'item_id', 'time'], how='left').fillna(0);
df_X = df_X.join(df_X_uit3_today, on=['user_id', 'item_id', 'time'], how='left').fillna(0);
df_X = df_X.join(df_X_uit4_today, on=['user_id', 'item_id', 'time'], how='left').fillna(0);

#1 day
df_X = df_X.join(df_X_uit1_1day, on=['user_id', 'item_id', 'time'], how='left').fillna(0);
df_X = df_X.join(df_X_uit2_1day, on=['user_id', 'item_id', 'time'], how='left').fillna(0);
df_X = df_X.join(df_X_uit3_1day, on=['user_id', 'item_id', 'time'], how='left').fillna(0);
df_X = df_X.join(df_X_uit4_1day, on=['user_id', 'item_id', 'time'], how='left').fillna(0);

#2 day
df_X = df_X.join(df_X_uit1_2day, on=['user_id', 'item_id', 'time'], how='left').fillna(0);
df_X = df_X.join(df_X_uit2_2day, on=['user_id', 'item_id', 'time'], how='left').fillna(0);
df_X = df_X.join(df_X_uit3_2day, on=['user_id', 'item_id', 'time'], how='left').fillna(0);
df_X = df_X.join(df_X_uit4_2day, on=['user_id', 'item_id', 'time'], how='left').fillna(0);

#3-10 day
df_X = df_X.join(df_X_uit1_310day, on=['user_id', 'item_id', 'time'], how='left').fillna(0);
df_X = df_X.join(df_X_uit2_310day, on=['user_id', 'item_id', 'time'], how='left').fillna(0);
df_X = df_X.join(df_X_uit3_310day, on=['user_id', 'item_id', 'time'], how='left').fillna(0);
df_X = df_X.join(df_X_uit4_310day, on=['user_id', 'item_id', 'time'], how='left').fillna(0);
df_X = df_X.join(df_X_uit5_310day, on=['user_id', 'item_id', 'time'], how='left').fillna(0);
df_X = df_X.join(df_X_uit6_310day, on=['user_id', 'item_id', 'time'], how='left').fillna(0);
df_X = df_X.join(df_X_uit7_310day, on=['user_id', 'item_id', 'time'], how='left').fillna(0);
df_X = df_X.join(df_X_uit8_310day, on=['user_id', 'item_id', 'time'], how='left').fillna(0);

#0-10 day
df_X = df_X.join(df_X_uit1_010day, on=['user_id', 'item_id', 'time'], how='left').fillna(0);

#whole period
df_X = df_X.join(df_X_uit1, on=['user_id', 'item_id', 'time'], how='left').fillna(30);
df_X = df_X.join(df_X_uit2, on=['user_id', 'item_id', 'time'], how='left').fillna(30);

##X_Feature normalization/standardization(not need in decision tree model)

##objection extraction
df_y1 = df_clean.loc[df_clean.behavior_type==4];
df_y1.time = df_y1.time.sub(np.timedelta64(1, 'D'));
df_y1 = df_y1.groupby(['user_id', 'item_id', 'item_category', 'time']).size().reset_index(name='buy_nextday');
df_y1.buy_nextday = 1;
df_y1 = df_y1.set_index(['user_id', 'item_id', 'item_category', 'time']);

df_y2 = df_clean.loc[df_clean.behavior_type==4];
df_y2.time = df_y2.time.sub(np.timedelta64(2, 'D'));
df_y2 = df_y2.groupby(['user_id', 'item_id', 'item_category', 'time']).size().reset_index(name='buy_next2day');
df_y2.buy_next2day = 1;
df_y2 = df_y2.set_index(['user_id', 'item_id', 'item_category', 'time']);

##build Xy
#join y1, y2
df_Xy = df_X.join(df_y1, on=['user_id', 'item_id', 'item_category', 'time'], how='left').fillna(0);
df_Xy = df_Xy.join(df_y2, on=['user_id', 'item_id', 'item_category', 'time'], how='left').fillna(0);

##save Xy
df_Xy.to_csv('df_Xy.csv', index=False);

