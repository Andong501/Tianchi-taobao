##Tianchi competition: Taobao purchase prediction version 2

import matplotlib as mlt
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
import numpy as np 
import pandas as pd 
import datetime
import warnings
from patsy import dmatrices
from sklearn.preprocessing import MinMaxScaler
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

##clean data
df1.drop(['user_geohash'], axis=1, inplace=True);
#item-cate subset
df2.drop(['item_geohash'], axis=1, inplace=True);
df2.drop_duplicates(inplace=True);
#item-cate allset
df3 = df1.loc[:, ['item_id', 'item_category']].drop_duplicates();

#allset
df_clean_all = pd.concat([df1, pd.DataFrame({'date': pd.to_datetime(df1.time).values.astype('datetime64[D]')})], axis=1);
df_clean_all = df_clean_all.loc[df_clean_all.date>np.datetime64('2014-12-07')];
#subset
df_clean_sub = df_clean_all.join(df2.drop(['item_category'], axis=1, inplace=False).set_index('item_id'), on='item_id', how='inner');

##construct feature columns
def feature_builder(bh_type, index_type, index_list, periods, ifmv):
    temp0 = df_clean_sub.loc[df_clean_sub.behavior_type==bh_type].groupby(index_list).size().reset_index(name='0day');
    temp_join = temp0.set_index(index_list);
    bh_name = ['ck', 'cl', 'ct', 'by'][bh_type-1];
    index_str = ['u_', 'c_', 'i_', 'uc_', 'ui_'][index_type-1];
    for i in range(1, periods):
	    tempi = temp0.rename(columns={'0day': str(i)+'day'});
	    tempi.date = tempi.date.add(np.timedelta64(i, 'D'));
	    tempi = tempi.loc[tempi.date<=np.datetime64('2014-12-18')];
	    temp_join = temp_join.join(tempi.set_index(index_list), how='outer').fillna(0);
    #mean & variance in past period
    if ifmv == 1:
    	kwargs = {index_str+bh_name+'_mean': temp_join.replace(0, np.NaN).mean(axis=1, skipna=True), index_str+bh_name+'_variance': temp_join.replace(0, np.NaN).var(axis=1, skipna=True)};
    	temp_join = temp_join.assign(**kwargs);
    else:
    	pass;
    #bh counts in some periods
    for i in [1, 3, 7]:
	    kwargs = {index_str+bh_name+'_'+str(i)+'day': temp_join.iloc[:, 0:i].sum(axis=1)};
	    temp_join = temp_join.assign(**kwargs);
    temp_join.drop(temp_join.columns[0:7], axis=1, inplace=True);
    temp_join.fillna(0, inplace=True);
    return temp_join;

#user#
df_X_u_ck = feature_builder(1, 1, ['user_id', 'date'], 7, 1);
df_X_u_cl = feature_builder(2, 1, ['user_id', 'date'], 7, 1);
df_X_u_ct = feature_builder(3, 1, ['user_id', 'date'], 7, 1);
df_X_u_by = feature_builder(4, 1, ['user_id', 'date'], 7, 1);
df_X_u_join = df_X_u_ck.join(df_X_u_cl, how='outer').join(df_X_u_ct, how='outer').join(df_X_u_by, how='outer').fillna(0);
df_X_u_join = df_X_u_join.assign(u_bh_cvr_1day=df_X_u_join.u_by_1day/(df_X_u_join.u_ck_1day+df_X_u_join.u_cl_1day+df_X_u_join.u_ct_1day)).fillna(0).replace(np.inf, 0);
df_X_u_join = df_X_u_join.assign(u_ck_cvr_1day=df_X_u_join.u_by_1day/df_X_u_join.u_ck_1day).fillna(0).replace(np.inf, 0);
df_X_u_join = df_X_u_join.assign(u_cl_cvr_1day=df_X_u_join.u_by_1day/df_X_u_join.u_cl_1day).fillna(0).replace(np.inf, 0);
df_X_u_join = df_X_u_join.assign(u_ct_cvr_1day=df_X_u_join.u_by_1day/df_X_u_join.u_ct_1day).fillna(0).replace(np.inf, 0);
df_X_u_join = df_X_u_join.assign(u_active_1day=df_X_u_join.u_ck_1day*0.1+df_X_u_join.u_cl_1day*1+df_X_u_join.u_ct_1day*2+df_X_u_join.u_by_1day*5);
df_X_u_join = df_X_u_join.assign(u_bh_cvr_3day=df_X_u_join.u_by_3day/(df_X_u_join.u_ck_3day+df_X_u_join.u_cl_3day+df_X_u_join.u_ct_3day)).fillna(0).replace(np.inf, 0);
df_X_u_join = df_X_u_join.assign(u_ck_cvr_3day=df_X_u_join.u_by_3day/df_X_u_join.u_ck_3day).fillna(0).replace(np.inf, 0);
df_X_u_join = df_X_u_join.assign(u_cl_cvr_3day=df_X_u_join.u_by_3day/df_X_u_join.u_cl_3day).fillna(0).replace(np.inf, 0);
df_X_u_join = df_X_u_join.assign(u_ct_cvr_3day=df_X_u_join.u_by_3day/df_X_u_join.u_ct_3day).fillna(0).replace(np.inf, 0);
df_X_u_join = df_X_u_join.assign(u_active_3day=df_X_u_join.u_ck_3day*0.1+df_X_u_join.u_cl_3day*1+df_X_u_join.u_ct_3day*2+df_X_u_join.u_by_3day*5);
df_X_u_join = df_X_u_join.assign(u_bh_cvr_7day=df_X_u_join.u_by_7day/(df_X_u_join.u_ck_7day+df_X_u_join.u_cl_7day+df_X_u_join.u_ct_7day)).fillna(0).replace(np.inf, 0);
df_X_u_join = df_X_u_join.assign(u_ck_cvr_7day=df_X_u_join.u_by_7day/df_X_u_join.u_ck_7day).fillna(0).replace(np.inf, 0);
df_X_u_join = df_X_u_join.assign(u_cl_cvr_7day=df_X_u_join.u_by_7day/df_X_u_join.u_cl_7day).fillna(0).replace(np.inf, 0);
df_X_u_join = df_X_u_join.assign(u_ct_cvr_7day=df_X_u_join.u_by_7day/df_X_u_join.u_ct_7day).fillna(0).replace(np.inf, 0);
df_X_u_join = df_X_u_join.assign(u_active_7day=df_X_u_join.u_ck_7day*0.1+df_X_u_join.u_cl_7day*1+df_X_u_join.u_ct_7day*2+df_X_u_join.u_by_7day*5);#

#category#
df_X_c_ck = feature_builder(1, 2, ['item_category', 'date'], 7, 1);
df_X_c_cl = feature_builder(2, 2, ['item_category', 'date'], 7, 1);
df_X_c_ct = feature_builder(3, 2, ['item_category', 'date'], 7, 1);
df_X_c_by = feature_builder(4, 2, ['item_category', 'date'], 7, 1);
df_X_c_join = df_X_c_ck.join(df_X_c_cl, how='outer').join(df_X_c_ct, how='outer').join(df_X_c_by, how='outer').fillna(0);
df_X_c_join = df_X_c_join.assign(c_bh_cvr_1day=df_X_c_join.c_by_1day/(df_X_c_join.c_ck_1day+df_X_c_join.c_cl_1day+df_X_c_join.c_ct_1day)).fillna(0).replace(np.inf, 0);
df_X_c_join = df_X_c_join.assign(c_ck_cvr_1day=df_X_c_join.c_by_1day/df_X_c_join.c_ck_1day).fillna(0).replace(np.inf, 0);
df_X_c_join = df_X_c_join.assign(c_cl_cvr_1day=df_X_c_join.c_by_1day/df_X_c_join.c_cl_1day).fillna(0).replace(np.inf, 0);
df_X_c_join = df_X_c_join.assign(c_ct_cvr_1day=df_X_c_join.c_by_1day/df_X_c_join.c_ct_1day).fillna(0).replace(np.inf, 0);
df_X_c_join = df_X_c_join.assign(c_active_1day=df_X_c_join.c_ck_1day*0.1+df_X_c_join.c_cl_1day*1+df_X_c_join.c_ct_1day*2+df_X_c_join.c_by_1day*5);
df_X_c_join = df_X_c_join.assign(c_bh_cvr_3day=df_X_c_join.c_by_3day/(df_X_c_join.c_ck_3day+df_X_c_join.c_cl_3day+df_X_c_join.c_ct_3day)).fillna(0).replace(np.inf, 0);
df_X_c_join = df_X_c_join.assign(c_ck_cvr_3day=df_X_c_join.c_by_3day/df_X_c_join.c_ck_3day).fillna(0).replace(np.inf, 0);
df_X_c_join = df_X_c_join.assign(c_cl_cvr_3day=df_X_c_join.c_by_3day/df_X_c_join.c_cl_3day).fillna(0).replace(np.inf, 0);
df_X_c_join = df_X_c_join.assign(c_ct_cvr_3day=df_X_c_join.c_by_3day/df_X_c_join.c_ct_3day).fillna(0).replace(np.inf, 0);
df_X_c_join = df_X_c_join.assign(c_active_3day=df_X_c_join.c_ck_3day*0.1+df_X_c_join.c_cl_3day*1+df_X_c_join.c_ct_3day*2+df_X_c_join.c_by_3day*5);
df_X_c_join = df_X_c_join.assign(c_bh_cvr_7day=df_X_c_join.c_by_7day/(df_X_c_join.c_ck_7day+df_X_c_join.c_cl_7day+df_X_c_join.c_ct_7day)).fillna(0).replace(np.inf, 0);
df_X_c_join = df_X_c_join.assign(c_ck_cvr_7day=df_X_c_join.c_by_7day/df_X_c_join.c_ck_7day).fillna(0).replace(np.inf, 0);
df_X_c_join = df_X_c_join.assign(c_cl_cvr_7day=df_X_c_join.c_by_7day/df_X_c_join.c_cl_7day).fillna(0).replace(np.inf, 0);
df_X_c_join = df_X_c_join.assign(c_ct_cvr_7day=df_X_c_join.c_by_7day/df_X_c_join.c_ct_7day).fillna(0).replace(np.inf, 0);
df_X_c_join = df_X_c_join.assign(c_active_7day=df_X_c_join.c_ck_7day*0.1+df_X_c_join.c_cl_7day*1+df_X_c_join.c_ct_7day*2+df_X_c_join.c_by_7day*5);#

#item#
df_X_i_ck = feature_builder(1, 3, ['item_id', 'date'], 7, 1);
df_X_i_cl = feature_builder(2, 3, ['item_id', 'date'], 7, 1);
df_X_i_ct = feature_builder(3, 3, ['item_id', 'date'], 7, 1);
df_X_i_by = feature_builder(4, 3, ['item_id', 'date'], 7, 1);
df_X_i_join = df_X_i_ck.join(df_X_i_cl, how='outer').join(df_X_i_ct, how='outer').join(df_X_i_by, how='outer').fillna(0);
df_X_i_join = df_X_i_join.assign(i_bh_cvr_1day=df_X_i_join.i_by_1day/(df_X_i_join.i_ck_1day+df_X_i_join.i_cl_1day+df_X_i_join.i_ct_1day)).fillna(0).replace(np.inf, 0);
df_X_i_join = df_X_i_join.assign(i_ck_cvr_1day=df_X_i_join.i_by_1day/df_X_i_join.i_ck_1day).fillna(0).replace(np.inf, 0);
df_X_i_join = df_X_i_join.assign(i_cl_cvr_1day=df_X_i_join.i_by_1day/df_X_i_join.i_cl_1day).fillna(0).replace(np.inf, 0);
df_X_i_join = df_X_i_join.assign(i_ct_cvr_1day=df_X_i_join.i_by_1day/df_X_i_join.i_ct_1day).fillna(0).replace(np.inf, 0);
df_X_i_join = df_X_i_join.assign(i_active_1day=df_X_i_join.i_ck_1day*0.1+df_X_i_join.i_cl_1day*1+df_X_i_join.i_ct_1day*2+df_X_i_join.i_by_1day*5);
df_X_i_join = df_X_i_join.assign(i_bh_cvr_3day=df_X_i_join.i_by_3day/(df_X_i_join.i_ck_3day+df_X_i_join.i_cl_3day+df_X_i_join.i_ct_3day)).fillna(0).replace(np.inf, 0);
df_X_i_join = df_X_i_join.assign(i_ck_cvr_3day=df_X_i_join.i_by_3day/df_X_i_join.i_ck_3day).fillna(0).replace(np.inf, 0);
df_X_i_join = df_X_i_join.assign(i_cl_cvr_3day=df_X_i_join.i_by_3day/df_X_i_join.i_cl_3day).fillna(0).replace(np.inf, 0);
df_X_i_join = df_X_i_join.assign(i_ct_cvr_3day=df_X_i_join.i_by_3day/df_X_i_join.i_ct_3day).fillna(0).replace(np.inf, 0);
df_X_i_join = df_X_i_join.assign(i_active_3day=df_X_i_join.i_ck_3day*0.1+df_X_i_join.i_cl_3day*1+df_X_i_join.i_ct_3day*2+df_X_i_join.i_by_3day*5);
df_X_i_join = df_X_i_join.assign(i_bh_cvr_7day=df_X_i_join.i_by_7day/(df_X_i_join.i_ck_7day+df_X_i_join.i_cl_7day+df_X_i_join.i_ct_7day)).fillna(0).replace(np.inf, 0);
df_X_i_join = df_X_i_join.assign(i_ck_cvr_7day=df_X_i_join.i_by_7day/df_X_i_join.i_ck_7day).fillna(0).replace(np.inf, 0);
df_X_i_join = df_X_i_join.assign(i_cl_cvr_7day=df_X_i_join.i_by_7day/df_X_i_join.i_cl_7day).fillna(0).replace(np.inf, 0);
df_X_i_join = df_X_i_join.assign(i_ct_cvr_7day=df_X_i_join.i_by_7day/df_X_i_join.i_ct_7day).fillna(0).replace(np.inf, 0);
df_X_i_join = df_X_i_join.assign(i_active_7day=df_X_i_join.i_ck_7day*0.1+df_X_i_join.i_cl_7day*1+df_X_i_join.i_ct_7day*2+df_X_i_join.i_by_7day*5);#

ic_pr_list = ['i_ck_1day', 'i_cl_1day', 'i_ct_1day', 'i_by_1day', 'i_ck_3day', 'i_cl_3day', 'i_ct_3day', 'i_by_3day', 'i_ck_7day', 'i_cl_7day', 'i_ct_7day', 'i_by_7day', \
'c_ck_1day', 'c_cl_1day', 'c_ct_1day', 'c_by_1day', 'c_ck_3day', 'c_cl_3day', 'c_ct_3day', 'c_by_3day', 'c_ck_7day', 'c_cl_7day', 'c_ct_7day', 'c_by_7day', 'item_id'];
df_X_ic_join0 = df_X_i_join.reset_index(level=[1]).join(df2.set_index('item_id'), how='left').reset_index(level=[0]).set_index(['item_category', 'date']).join(df_X_c_join, how='left');
df_X_ic_join1 = df_X_ic_join0.loc[:, ic_pr_list];
df_X_ic_join1 = df_X_ic_join1.assign(i_ck_pr_1day=df_X_ic_join1.i_ck_1day/df_X_ic_join1.c_ck_1day).fillna(0);
df_X_ic_join1 = df_X_ic_join1.assign(i_cl_pr_1day=df_X_ic_join1.i_cl_1day/df_X_ic_join1.c_cl_1day).fillna(0);
df_X_ic_join1 = df_X_ic_join1.assign(i_ct_pr_1day=df_X_ic_join1.i_ct_1day/df_X_ic_join1.c_ct_1day).fillna(0);
df_X_ic_join1 = df_X_ic_join1.assign(i_by_pr_1day=df_X_ic_join1.i_by_1day/df_X_ic_join1.c_by_1day).fillna(0);
df_X_ic_join1 = df_X_ic_join1.assign(i_ck_pr_3day=df_X_ic_join1.i_ck_3day/df_X_ic_join1.c_ck_3day).fillna(0);
df_X_ic_join1 = df_X_ic_join1.assign(i_cl_pr_3day=df_X_ic_join1.i_cl_3day/df_X_ic_join1.c_cl_3day).fillna(0);
df_X_ic_join1 = df_X_ic_join1.assign(i_ct_pr_3day=df_X_ic_join1.i_ct_3day/df_X_ic_join1.c_ct_3day).fillna(0);
df_X_ic_join1 = df_X_ic_join1.assign(i_by_pr_3day=df_X_ic_join1.i_by_3day/df_X_ic_join1.c_by_3day).fillna(0);
df_X_ic_join1 = df_X_ic_join1.assign(i_ck_pr_7day=df_X_ic_join1.i_ck_7day/df_X_ic_join1.c_ck_7day).fillna(0);
df_X_ic_join1 = df_X_ic_join1.assign(i_cl_pr_7day=df_X_ic_join1.i_cl_7day/df_X_ic_join1.c_cl_7day).fillna(0);
df_X_ic_join1 = df_X_ic_join1.assign(i_ct_pr_7day=df_X_ic_join1.i_ct_7day/df_X_ic_join1.c_ct_7day).fillna(0);
df_X_ic_join1 = df_X_ic_join1.assign(i_by_pr_7day=df_X_ic_join1.i_by_7day/df_X_ic_join1.c_by_7day).fillna(0);

df_X_ic_join2 = df_X_ic_join0.loc[:, ic_pr_list[0:12]];
df_X_ic_join2 = df_X_ic_join2.reset_index(level=[0, 1]).groupby(['item_category', 'date']).mean();

df_X_ic_join = df_X_ic_join1.join(df_X_ic_join2, rsuffix='_mean', how='left');
df_X_ic_join = df_X_ic_join.assign(i_ck_dv_1day=df_X_ic_join.i_ck_1day-df_X_ic_join.i_ck_1day_mean);
df_X_ic_join = df_X_ic_join.assign(i_cl_dv_1day=df_X_ic_join.i_cl_1day-df_X_ic_join.i_cl_1day_mean);
df_X_ic_join = df_X_ic_join.assign(i_ct_dv_1day=df_X_ic_join.i_ct_1day-df_X_ic_join.i_ct_1day_mean);
df_X_ic_join = df_X_ic_join.assign(i_by_dv_1day=df_X_ic_join.i_by_1day-df_X_ic_join.i_by_1day_mean);
df_X_ic_join = df_X_ic_join.assign(i_ck_dv_3day=df_X_ic_join.i_ck_3day-df_X_ic_join.i_ck_3day_mean);
df_X_ic_join = df_X_ic_join.assign(i_cl_dv_3day=df_X_ic_join.i_cl_3day-df_X_ic_join.i_cl_3day_mean);
df_X_ic_join = df_X_ic_join.assign(i_ct_dv_3day=df_X_ic_join.i_ct_3day-df_X_ic_join.i_ct_3day_mean);
df_X_ic_join = df_X_ic_join.assign(i_by_dv_3day=df_X_ic_join.i_by_3day-df_X_ic_join.i_by_3day_mean);
df_X_ic_join = df_X_ic_join.assign(i_ck_dv_7day=df_X_ic_join.i_ck_7day-df_X_ic_join.i_ck_7day_mean);
df_X_ic_join = df_X_ic_join.assign(i_cl_dv_7day=df_X_ic_join.i_cl_7day-df_X_ic_join.i_cl_7day_mean);
df_X_ic_join = df_X_ic_join.assign(i_ct_dv_7day=df_X_ic_join.i_ct_7day-df_X_ic_join.i_ct_7day_mean);
df_X_ic_join = df_X_ic_join.assign(i_by_dv_7day=df_X_ic_join.i_by_7day-df_X_ic_join.i_by_7day_mean);
df_X_ic_join = df_X_ic_join.set_index(['item_id'], append=True).reorder_levels(['item_id', 'item_category', 'date']);
df_X_ic_join.drop(df_X_ic_join.ix[:, 0:24].head(0).columns, axis=1, inplace=True);
df_X_ic_join.drop(df_X_ic_join.ix[:, 12:24].head(0).columns, axis=1, inplace=True);#

#user & category#
df_X_uc_ck = feature_builder(1, 4, ['user_id', 'item_category', 'date'], 7, 0);
df_X_uc_cl = feature_builder(2, 4, ['user_id', 'item_category', 'date'], 7, 0);
df_X_uc_ct = feature_builder(3, 4, ['user_id', 'item_category', 'date'], 7, 0);
df_X_uc_by = feature_builder(4, 4, ['user_id', 'item_category', 'date'], 7, 0);
df_X_uc_join = df_X_uc_ck.join(df_X_uc_cl, how='outer').join(df_X_uc_ct, how='outer').join(df_X_uc_by, how='outer').fillna(0);#

uc_pr1_list = ['u_ck_1day', 'u_cl_1day', 'u_ct_1day', 'u_by_1day', 'u_ck_3day', 'u_cl_3day', 'u_ct_3day', 'u_by_3day', 'u_ck_7day', 'u_cl_7day', 'u_ct_7day', 'u_by_7day'];
df_X_uc_join1 = df_X_uc_join.reset_index(level=[1]);
df_X_uc_join1 = df_X_uc_join1.join(df_X_u_join.loc[:, uc_pr1_list], how='left');
df_X_uc_join1 = df_X_uc_join1.assign(uc_ck_pr1_1day=df_X_uc_join1.uc_ck_1day/df_X_uc_join1.u_ck_1day).fillna(0);
df_X_uc_join1 = df_X_uc_join1.assign(uc_cl_pr1_1day=df_X_uc_join1.uc_cl_1day/df_X_uc_join1.u_cl_1day).fillna(0);
df_X_uc_join1 = df_X_uc_join1.assign(uc_ct_pr1_1day=df_X_uc_join1.uc_ct_1day/df_X_uc_join1.u_ct_1day).fillna(0);
df_X_uc_join1 = df_X_uc_join1.assign(uc_by_pr1_1day=df_X_uc_join1.uc_by_1day/df_X_uc_join1.u_by_1day).fillna(0);
df_X_uc_join1 = df_X_uc_join1.assign(uc_ck_pr1_3day=df_X_uc_join1.uc_ck_3day/df_X_uc_join1.u_ck_3day).fillna(0);
df_X_uc_join1 = df_X_uc_join1.assign(uc_cl_pr1_3day=df_X_uc_join1.uc_cl_3day/df_X_uc_join1.u_cl_3day).fillna(0);
df_X_uc_join1 = df_X_uc_join1.assign(uc_ct_pr1_3day=df_X_uc_join1.uc_ct_3day/df_X_uc_join1.u_ct_3day).fillna(0);
df_X_uc_join1 = df_X_uc_join1.assign(uc_by_pr1_3day=df_X_uc_join1.uc_by_3day/df_X_uc_join1.u_by_3day).fillna(0);
df_X_uc_join1 = df_X_uc_join1.assign(uc_ck_pr1_7day=df_X_uc_join1.uc_ck_7day/df_X_uc_join1.u_ck_7day).fillna(0);
df_X_uc_join1 = df_X_uc_join1.assign(uc_cl_pr1_7day=df_X_uc_join1.uc_cl_7day/df_X_uc_join1.u_cl_7day).fillna(0);
df_X_uc_join1 = df_X_uc_join1.assign(uc_ct_pr1_7day=df_X_uc_join1.uc_ct_7day/df_X_uc_join1.u_ct_7day).fillna(0);
df_X_uc_join1 = df_X_uc_join1.assign(uc_by_pr1_7day=df_X_uc_join1.uc_by_7day/df_X_uc_join1.u_by_7day).fillna(0);

df_X_uc_join2 = df_X_uc_join.reset_index(level=[0, 1, 2]);
df_X_uc_join2 = df_X_uc_join2.groupby(['user_id', 'date']).mean();

df_X_uc_join1 = df_X_uc_join1.join(df_X_uc_join2.drop(['item_category'], axis=1), rsuffix='_mean', how='left');
df_X_uc_join1 = df_X_uc_join1.assign(uc_ck_dv1_1day=df_X_uc_join1.uc_ck_1day-df_X_uc_join1.uc_ck_1day_mean);
df_X_uc_join1 = df_X_uc_join1.assign(uc_cl_dv1_1day=df_X_uc_join1.uc_cl_1day-df_X_uc_join1.uc_cl_1day_mean);
df_X_uc_join1 = df_X_uc_join1.assign(uc_ct_dv1_1day=df_X_uc_join1.uc_ct_1day-df_X_uc_join1.uc_ct_1day_mean);
df_X_uc_join1 = df_X_uc_join1.assign(uc_by_dv1_1day=df_X_uc_join1.uc_by_1day-df_X_uc_join1.uc_by_1day_mean);
df_X_uc_join1 = df_X_uc_join1.assign(uc_ck_dv1_3day=df_X_uc_join1.uc_ck_3day-df_X_uc_join1.uc_ck_3day_mean);
df_X_uc_join1 = df_X_uc_join1.assign(uc_cl_dv1_3day=df_X_uc_join1.uc_cl_3day-df_X_uc_join1.uc_cl_3day_mean);
df_X_uc_join1 = df_X_uc_join1.assign(uc_ct_dv1_3day=df_X_uc_join1.uc_ct_3day-df_X_uc_join1.uc_ct_3day_mean);
df_X_uc_join1 = df_X_uc_join1.assign(uc_by_dv1_3day=df_X_uc_join1.uc_by_3day-df_X_uc_join1.uc_by_3day_mean);
df_X_uc_join1 = df_X_uc_join1.assign(uc_ck_dv1_7day=df_X_uc_join1.uc_ck_7day-df_X_uc_join1.uc_ck_7day_mean);
df_X_uc_join1 = df_X_uc_join1.assign(uc_cl_dv1_7day=df_X_uc_join1.uc_cl_7day-df_X_uc_join1.uc_cl_7day_mean);
df_X_uc_join1 = df_X_uc_join1.assign(uc_ct_dv1_7day=df_X_uc_join1.uc_ct_7day-df_X_uc_join1.uc_ct_7day_mean);
df_X_uc_join1 = df_X_uc_join1.assign(uc_by_dv1_7day=df_X_uc_join1.uc_by_7day-df_X_uc_join1.uc_by_7day_mean);
df_X_uc_join1 = df_X_uc_join1.set_index(['item_category'], append=True).reorder_levels(['user_id', 'item_category', 'date']);
df_X_uc_join1.drop(df_X_uc_join1.ix[:, 0:24].head(0).columns, axis=1, inplace=True);
df_X_uc_join1.drop(df_X_uc_join1.ix[:, 12:24].head(0).columns, axis=1, inplace=True);#

uc_pr2_list = ['c_ck_1day', 'c_cl_1day', 'c_ct_1day', 'c_by_1day', 'c_ck_3day', 'c_cl_3day', 'c_ct_3day', 'c_by_3day', 'c_ck_7day', 'c_cl_7day', 'c_ct_7day', 'c_by_7day'];
df_X_uc_join3 = df_X_uc_join.reset_index(level=[0]);
df_X_uc_join3 = df_X_uc_join3.join(df_X_c_join.loc[:, uc_pr2_list], how='left');
df_X_uc_join3 = df_X_uc_join3.assign(uc_ck_pr2_1day=df_X_uc_join3.uc_ck_1day/df_X_uc_join3.c_ck_1day).fillna(0);
df_X_uc_join3 = df_X_uc_join3.assign(uc_cl_pr2_1day=df_X_uc_join3.uc_cl_1day/df_X_uc_join3.c_cl_1day).fillna(0);
df_X_uc_join3 = df_X_uc_join3.assign(uc_ct_pr2_1day=df_X_uc_join3.uc_ct_1day/df_X_uc_join3.c_ct_1day).fillna(0);
df_X_uc_join3 = df_X_uc_join3.assign(uc_by_pr2_1day=df_X_uc_join3.uc_by_1day/df_X_uc_join3.c_by_1day).fillna(0);
df_X_uc_join3 = df_X_uc_join3.assign(uc_ck_pr2_3day=df_X_uc_join3.uc_ck_3day/df_X_uc_join3.c_ck_3day).fillna(0);
df_X_uc_join3 = df_X_uc_join3.assign(uc_cl_pr2_3day=df_X_uc_join3.uc_cl_3day/df_X_uc_join3.c_cl_3day).fillna(0);
df_X_uc_join3 = df_X_uc_join3.assign(uc_ct_pr2_3day=df_X_uc_join3.uc_ct_3day/df_X_uc_join3.c_ct_3day).fillna(0);
df_X_uc_join3 = df_X_uc_join3.assign(uc_by_pr2_3day=df_X_uc_join3.uc_by_3day/df_X_uc_join3.c_by_3day).fillna(0);
df_X_uc_join3 = df_X_uc_join3.assign(uc_ck_pr2_7day=df_X_uc_join3.uc_ck_7day/df_X_uc_join3.c_ck_7day).fillna(0);
df_X_uc_join3 = df_X_uc_join3.assign(uc_cl_pr2_7day=df_X_uc_join3.uc_cl_7day/df_X_uc_join3.c_cl_7day).fillna(0);
df_X_uc_join3 = df_X_uc_join3.assign(uc_ct_pr2_7day=df_X_uc_join3.uc_ct_7day/df_X_uc_join3.c_ct_7day).fillna(0);
df_X_uc_join3 = df_X_uc_join3.assign(uc_by_pr2_7day=df_X_uc_join3.uc_by_7day/df_X_uc_join3.c_by_7day).fillna(0);

df_X_uc_join4 = df_X_uc_join.reset_index(level=[0, 1, 2]);
df_X_uc_join4 = df_X_uc_join4.groupby(['item_category', 'date']).mean();

df_X_uc_join3 = df_X_uc_join3.join(df_X_uc_join4.drop(['user_id'], axis=1), rsuffix='_mean', how='left');
df_X_uc_join3 = df_X_uc_join3.assign(uc_ck_dv2_1day=df_X_uc_join3.uc_ck_1day-df_X_uc_join3.uc_ck_1day_mean);
df_X_uc_join3 = df_X_uc_join3.assign(uc_cl_dv2_1day=df_X_uc_join3.uc_cl_1day-df_X_uc_join3.uc_cl_1day_mean);
df_X_uc_join3 = df_X_uc_join3.assign(uc_ct_dv2_1day=df_X_uc_join3.uc_ct_1day-df_X_uc_join3.uc_ct_1day_mean);
df_X_uc_join3 = df_X_uc_join3.assign(uc_by_dv2_1day=df_X_uc_join3.uc_by_1day-df_X_uc_join3.uc_by_1day_mean);
df_X_uc_join3 = df_X_uc_join3.assign(uc_ck_dv2_3day=df_X_uc_join3.uc_ck_3day-df_X_uc_join3.uc_ck_3day_mean);
df_X_uc_join3 = df_X_uc_join3.assign(uc_cl_dv2_3day=df_X_uc_join3.uc_cl_3day-df_X_uc_join3.uc_cl_3day_mean);
df_X_uc_join3 = df_X_uc_join3.assign(uc_ct_dv2_3day=df_X_uc_join3.uc_ct_3day-df_X_uc_join3.uc_ct_3day_mean);
df_X_uc_join3 = df_X_uc_join3.assign(uc_by_dv2_3day=df_X_uc_join3.uc_by_3day-df_X_uc_join3.uc_by_3day_mean);
df_X_uc_join3 = df_X_uc_join3.assign(uc_ck_dv2_7day=df_X_uc_join3.uc_ck_7day-df_X_uc_join3.uc_ck_7day_mean);
df_X_uc_join3 = df_X_uc_join3.assign(uc_cl_dv2_7day=df_X_uc_join3.uc_cl_7day-df_X_uc_join3.uc_cl_7day_mean);
df_X_uc_join3 = df_X_uc_join3.assign(uc_ct_dv2_7day=df_X_uc_join3.uc_ct_7day-df_X_uc_join3.uc_ct_7day_mean);
df_X_uc_join3 = df_X_uc_join3.assign(uc_by_dv2_7day=df_X_uc_join3.uc_by_7day-df_X_uc_join3.uc_by_7day_mean);
df_X_uc_join3 = df_X_uc_join3.set_index(['user_id'], append=True).reorder_levels(['user_id', 'item_category', 'date']);
df_X_uc_join3.drop(df_X_uc_join3.ix[:, 0:24].head(0).columns, axis=1, inplace=True);
df_X_uc_join3.drop(df_X_uc_join3.ix[:, 12:24].head(0).columns, axis=1, inplace=True);#

#user & item#
df_X_ui_ck = feature_builder(1, 5, ['user_id', 'item_id', 'date'], 7, 0);
df_X_ui_cl = feature_builder(2, 5, ['user_id', 'item_id', 'date'], 7, 0);
df_X_ui_ct = feature_builder(3, 5, ['user_id', 'item_id', 'date'], 7, 0);
df_X_ui_by = feature_builder(4, 5, ['user_id', 'item_id', 'date'], 7, 0);
df_X_ui_join = df_X_ui_ck.join(df_X_ui_cl, how='outer').join(df_X_ui_ct, how='outer').join(df_X_ui_by, how='outer').fillna(0);#

ui_pr1_list = ['u_ck_1day', 'u_cl_1day', 'u_ct_1day', 'u_by_1day', 'u_ck_3day', 'u_cl_3day', 'u_ct_3day', 'u_by_3day', 'u_ck_7day', 'u_cl_7day', 'u_ct_7day', 'u_by_7day'];
df_X_ui_join1 = df_X_ui_join.reset_index(level=[1]);
df_X_ui_join1 = df_X_ui_join1.join(df_X_u_join.loc[:, ui_pr1_list], how='left');
df_X_ui_join1 = df_X_ui_join1.assign(ui_ck_pr1_1day=df_X_ui_join1.ui_ck_1day/df_X_ui_join1.u_ck_1day).fillna(0);
df_X_ui_join1 = df_X_ui_join1.assign(ui_cl_pr1_1day=df_X_ui_join1.ui_cl_1day/df_X_ui_join1.u_cl_1day).fillna(0);
df_X_ui_join1 = df_X_ui_join1.assign(ui_ct_pr1_1day=df_X_ui_join1.ui_ct_1day/df_X_ui_join1.u_ct_1day).fillna(0);
df_X_ui_join1 = df_X_ui_join1.assign(ui_by_pr1_1day=df_X_ui_join1.ui_by_1day/df_X_ui_join1.u_by_1day).fillna(0);
df_X_ui_join1 = df_X_ui_join1.assign(ui_ck_pr1_3day=df_X_ui_join1.ui_ck_3day/df_X_ui_join1.u_ck_3day).fillna(0);
df_X_ui_join1 = df_X_ui_join1.assign(ui_cl_pr1_3day=df_X_ui_join1.ui_cl_3day/df_X_ui_join1.u_cl_3day).fillna(0);
df_X_ui_join1 = df_X_ui_join1.assign(ui_ct_pr1_3day=df_X_ui_join1.ui_ct_3day/df_X_ui_join1.u_ct_3day).fillna(0);
df_X_ui_join1 = df_X_ui_join1.assign(ui_by_pr1_3day=df_X_ui_join1.ui_by_3day/df_X_ui_join1.u_by_3day).fillna(0);
df_X_ui_join1 = df_X_ui_join1.assign(ui_ck_pr1_7day=df_X_ui_join1.ui_ck_7day/df_X_ui_join1.u_ck_7day).fillna(0);
df_X_ui_join1 = df_X_ui_join1.assign(ui_cl_pr1_7day=df_X_ui_join1.ui_cl_7day/df_X_ui_join1.u_cl_7day).fillna(0);
df_X_ui_join1 = df_X_ui_join1.assign(ui_ct_pr1_7day=df_X_ui_join1.ui_ct_7day/df_X_ui_join1.u_ct_7day).fillna(0);
df_X_ui_join1 = df_X_ui_join1.assign(ui_by_pr1_7day=df_X_ui_join1.ui_by_7day/df_X_ui_join1.u_by_7day).fillna(0);

df_X_ui_join2 = df_X_ui_join.reset_index(level=[0, 1, 2]);
df_X_ui_join2 = df_X_ui_join2.groupby(['user_id', 'date']).mean();

df_X_ui_join1 = df_X_ui_join1.join(df_X_ui_join2.drop(['item_id'], axis=1), rsuffix='_mean', how='left');
df_X_ui_join1 = df_X_ui_join1.assign(ui_ck_dv1_1day=df_X_ui_join1.ui_ck_1day-df_X_ui_join1.ui_ck_1day_mean);
df_X_ui_join1 = df_X_ui_join1.assign(ui_cl_dv1_1day=df_X_ui_join1.ui_cl_1day-df_X_ui_join1.ui_cl_1day_mean);
df_X_ui_join1 = df_X_ui_join1.assign(ui_ct_dv1_1day=df_X_ui_join1.ui_ct_1day-df_X_ui_join1.ui_ct_1day_mean);
df_X_ui_join1 = df_X_ui_join1.assign(ui_by_dv1_1day=df_X_ui_join1.ui_by_1day-df_X_ui_join1.ui_by_1day_mean);
df_X_ui_join1 = df_X_ui_join1.assign(ui_ck_dv1_3day=df_X_ui_join1.ui_ck_3day-df_X_ui_join1.ui_ck_3day_mean);
df_X_ui_join1 = df_X_ui_join1.assign(ui_cl_dv1_3day=df_X_ui_join1.ui_cl_3day-df_X_ui_join1.ui_cl_3day_mean);
df_X_ui_join1 = df_X_ui_join1.assign(ui_ct_dv1_3day=df_X_ui_join1.ui_ct_3day-df_X_ui_join1.ui_ct_3day_mean);
df_X_ui_join1 = df_X_ui_join1.assign(ui_by_dv1_3day=df_X_ui_join1.ui_by_3day-df_X_ui_join1.ui_by_3day_mean);
df_X_ui_join1 = df_X_ui_join1.assign(ui_ck_dv1_7day=df_X_ui_join1.ui_ck_7day-df_X_ui_join1.ui_ck_7day_mean);
df_X_ui_join1 = df_X_ui_join1.assign(ui_cl_dv1_7day=df_X_ui_join1.ui_cl_7day-df_X_ui_join1.ui_cl_7day_mean);
df_X_ui_join1 = df_X_ui_join1.assign(ui_ct_dv1_7day=df_X_ui_join1.ui_ct_7day-df_X_ui_join1.ui_ct_7day_mean);
df_X_ui_join1 = df_X_ui_join1.assign(ui_by_dv1_7day=df_X_ui_join1.ui_by_7day-df_X_ui_join1.ui_by_7day_mean);
df_X_ui_join1 = df_X_ui_join1.set_index(['item_id'], append=True).reorder_levels(['user_id', 'item_id', 'date']);
df_X_ui_join1.drop(df_X_ui_join1.ix[:, 0:24].head(0).columns, axis=1, inplace=True);
df_X_ui_join1.drop(df_X_ui_join1.ix[:, 12:24].head(0).columns, axis=1, inplace=True);#

ui_pr2_list = ['i_ck_1day', 'i_cl_1day', 'i_ct_1day', 'i_by_1day', 'i_ck_3day', 'i_cl_3day', 'i_ct_3day', 'i_by_3day', 'i_ck_7day', 'i_cl_7day', 'i_ct_7day', 'i_by_7day'];
df_X_ui_join3 = df_X_ui_join.reset_index(level=[0]);
df_X_ui_join3 = df_X_ui_join3.join(df_X_i_join.loc[:, ui_pr2_list], how='left');
df_X_ui_join3 = df_X_ui_join3.assign(ui_ck_pr2_1day=df_X_ui_join3.ui_ck_1day/df_X_ui_join3.i_ck_1day).fillna(0);
df_X_ui_join3 = df_X_ui_join3.assign(ui_cl_pr2_1day=df_X_ui_join3.ui_cl_1day/df_X_ui_join3.i_cl_1day).fillna(0);
df_X_ui_join3 = df_X_ui_join3.assign(ui_ct_pr2_1day=df_X_ui_join3.ui_ct_1day/df_X_ui_join3.i_ct_1day).fillna(0);
df_X_ui_join3 = df_X_ui_join3.assign(ui_by_pr2_1day=df_X_ui_join3.ui_by_1day/df_X_ui_join3.i_by_1day).fillna(0);
df_X_ui_join3 = df_X_ui_join3.assign(ui_ck_pr2_3day=df_X_ui_join3.ui_ck_3day/df_X_ui_join3.i_ck_3day).fillna(0);
df_X_ui_join3 = df_X_ui_join3.assign(ui_cl_pr2_3day=df_X_ui_join3.ui_cl_3day/df_X_ui_join3.i_cl_3day).fillna(0);
df_X_ui_join3 = df_X_ui_join3.assign(ui_ct_pr2_3day=df_X_ui_join3.ui_ct_3day/df_X_ui_join3.i_ct_3day).fillna(0);
df_X_ui_join3 = df_X_ui_join3.assign(ui_by_pr2_3day=df_X_ui_join3.ui_by_3day/df_X_ui_join3.i_by_3day).fillna(0);
df_X_ui_join3 = df_X_ui_join3.assign(ui_ck_pr2_7day=df_X_ui_join3.ui_ck_7day/df_X_ui_join3.i_ck_7day).fillna(0);
df_X_ui_join3 = df_X_ui_join3.assign(ui_cl_pr2_7day=df_X_ui_join3.ui_cl_7day/df_X_ui_join3.i_cl_7day).fillna(0);
df_X_ui_join3 = df_X_ui_join3.assign(ui_ct_pr2_7day=df_X_ui_join3.ui_ct_7day/df_X_ui_join3.i_ct_7day).fillna(0);
df_X_ui_join3 = df_X_ui_join3.assign(ui_by_pr2_7day=df_X_ui_join3.ui_by_7day/df_X_ui_join3.i_by_7day).fillna(0);

df_X_ui_join4 = df_X_ui_join.reset_index(level=[0, 1, 2]);
df_X_ui_join4 = df_X_ui_join4.groupby(['item_id', 'date']).mean();

df_X_ui_join3 = df_X_ui_join3.join(df_X_ui_join4.drop(['user_id'], axis=1), rsuffix='_mean', how='left');
df_X_ui_join3 = df_X_ui_join3.assign(ui_ck_dv2_1day=df_X_ui_join3.ui_ck_1day-df_X_ui_join3.ui_ck_1day_mean);
df_X_ui_join3 = df_X_ui_join3.assign(ui_cl_dv2_1day=df_X_ui_join3.ui_cl_1day-df_X_ui_join3.ui_cl_1day_mean);
df_X_ui_join3 = df_X_ui_join3.assign(ui_ct_dv2_1day=df_X_ui_join3.ui_ct_1day-df_X_ui_join3.ui_ct_1day_mean);
df_X_ui_join3 = df_X_ui_join3.assign(ui_by_dv2_1day=df_X_ui_join3.ui_by_1day-df_X_ui_join3.ui_by_1day_mean);
df_X_ui_join3 = df_X_ui_join3.assign(ui_ck_dv2_3day=df_X_ui_join3.ui_ck_3day-df_X_ui_join3.ui_ck_3day_mean);
df_X_ui_join3 = df_X_ui_join3.assign(ui_cl_dv2_3day=df_X_ui_join3.ui_cl_3day-df_X_ui_join3.ui_cl_3day_mean);
df_X_ui_join3 = df_X_ui_join3.assign(ui_ct_dv2_3day=df_X_ui_join3.ui_ct_3day-df_X_ui_join3.ui_ct_3day_mean);
df_X_ui_join3 = df_X_ui_join3.assign(ui_by_dv2_3day=df_X_ui_join3.ui_by_3day-df_X_ui_join3.ui_by_3day_mean);
df_X_ui_join3 = df_X_ui_join3.assign(ui_ck_dv2_7day=df_X_ui_join3.ui_ck_7day-df_X_ui_join3.ui_ck_7day_mean);
df_X_ui_join3 = df_X_ui_join3.assign(ui_cl_dv2_7day=df_X_ui_join3.ui_cl_7day-df_X_ui_join3.ui_cl_7day_mean);
df_X_ui_join3 = df_X_ui_join3.assign(ui_ct_dv2_7day=df_X_ui_join3.ui_ct_7day-df_X_ui_join3.ui_ct_7day_mean);
df_X_ui_join3 = df_X_ui_join3.assign(ui_by_dv2_7day=df_X_ui_join3.ui_by_7day-df_X_ui_join3.ui_by_7day_mean);
df_X_ui_join3 = df_X_ui_join3.set_index(['user_id'], append=True).reorder_levels(['user_id', 'item_id', 'date']);
df_X_ui_join3.drop(df_X_ui_join3.ix[:, 0:24].head(0).columns, axis=1, inplace=True);
df_X_ui_join3.drop(df_X_ui_join3.ix[:, 12:24].head(0).columns, axis=1, inplace=True);#

ui_pr3_list = ['ui_ck_1day', 'ui_cl_1day', 'ui_ct_1day', 'ui_by_1day', 'ui_ck_3day', 'ui_cl_3day', 'ui_ct_3day', 'ui_by_3day', 'ui_ck_7day', 'ui_cl_7day', 'ui_ct_7day', 'ui_by_7day', \
'uc_ck_1day', 'uc_cl_1day', 'uc_ct_1day', 'uc_by_1day', 'uc_ck_3day', 'uc_cl_3day', 'uc_ct_3day', 'uc_by_3day', 'uc_ck_7day', 'uc_cl_7day', 'uc_ct_7day', 'uc_by_7day', 'item_id'];
df_X_ui_join5 = df_X_ui_join.reset_index(level=[0, 2]).join(df2.set_index('item_id'), how='left').reset_index(level=[0]).set_index(['user_id', 'item_category', 'date']).join(df_X_uc_join, how='left');
df_X_ui_join6 = df_X_ui_join5.loc[:, ui_pr3_list];
df_X_ui_join6 = df_X_ui_join6.assign(ui_ck_pr3_1day=df_X_ui_join6.ui_ck_1day/df_X_ui_join6.uc_ck_1day).fillna(0);
df_X_ui_join6 = df_X_ui_join6.assign(ui_cl_pr3_1day=df_X_ui_join6.ui_cl_1day/df_X_ui_join6.uc_cl_1day).fillna(0);
df_X_ui_join6 = df_X_ui_join6.assign(ui_ct_pr3_1day=df_X_ui_join6.ui_ct_1day/df_X_ui_join6.uc_ct_1day).fillna(0);
df_X_ui_join6 = df_X_ui_join6.assign(ui_by_pr3_1day=df_X_ui_join6.ui_by_1day/df_X_ui_join6.uc_by_1day).fillna(0);
df_X_ui_join6 = df_X_ui_join6.assign(ui_ck_pr3_3day=df_X_ui_join6.ui_ck_3day/df_X_ui_join6.uc_ck_3day).fillna(0);
df_X_ui_join6 = df_X_ui_join6.assign(ui_cl_pr3_3day=df_X_ui_join6.ui_cl_3day/df_X_ui_join6.uc_cl_3day).fillna(0);
df_X_ui_join6 = df_X_ui_join6.assign(ui_ct_pr3_3day=df_X_ui_join6.ui_ct_3day/df_X_ui_join6.uc_ct_3day).fillna(0);
df_X_ui_join6 = df_X_ui_join6.assign(ui_by_pr3_3day=df_X_ui_join6.ui_by_3day/df_X_ui_join6.uc_by_3day).fillna(0);
df_X_ui_join6 = df_X_ui_join6.assign(ui_ck_pr3_7day=df_X_ui_join6.ui_ck_7day/df_X_ui_join6.uc_ck_7day).fillna(0);
df_X_ui_join6 = df_X_ui_join6.assign(ui_cl_pr3_7day=df_X_ui_join6.ui_cl_7day/df_X_ui_join6.uc_cl_7day).fillna(0);
df_X_ui_join6 = df_X_ui_join6.assign(ui_ct_pr3_7day=df_X_ui_join6.ui_ct_7day/df_X_ui_join6.uc_ct_7day).fillna(0);
df_X_ui_join6 = df_X_ui_join6.assign(ui_by_pr3_7day=df_X_ui_join6.ui_by_7day/df_X_ui_join6.uc_by_7day).fillna(0);

df_X_ui_join7 = df_X_ui_join5.loc[:, ui_pr3_list[0:12]];
df_X_ui_join7 = df_X_ui_join7.reset_index(level=[0, 1, 2]).groupby(['user_id', 'item_category', 'date']).mean();

df_X_ui_join8 = df_X_ui_join6.join(df_X_ui_join7, rsuffix='_mean', how='left');
df_X_ui_join8 = df_X_ui_join8.assign(ui_ck_dv3_1day=df_X_ui_join8.ui_ck_1day-df_X_ui_join8.ui_ck_1day_mean);
df_X_ui_join8 = df_X_ui_join8.assign(ui_cl_dv3_1day=df_X_ui_join8.ui_cl_1day-df_X_ui_join8.ui_cl_1day_mean);
df_X_ui_join8 = df_X_ui_join8.assign(ui_ct_dv3_1day=df_X_ui_join8.ui_ct_1day-df_X_ui_join8.ui_ct_1day_mean);
df_X_ui_join8 = df_X_ui_join8.assign(ui_by_dv3_1day=df_X_ui_join8.ui_by_1day-df_X_ui_join8.ui_by_1day_mean);
df_X_ui_join8 = df_X_ui_join8.assign(ui_ck_dv3_3day=df_X_ui_join8.ui_ck_3day-df_X_ui_join8.ui_ck_3day_mean);
df_X_ui_join8 = df_X_ui_join8.assign(ui_cl_dv3_3day=df_X_ui_join8.ui_cl_3day-df_X_ui_join8.ui_cl_3day_mean);
df_X_ui_join8 = df_X_ui_join8.assign(ui_ct_dv3_3day=df_X_ui_join8.ui_ct_3day-df_X_ui_join8.ui_ct_3day_mean);
df_X_ui_join8 = df_X_ui_join8.assign(ui_by_dv3_3day=df_X_ui_join8.ui_by_3day-df_X_ui_join8.ui_by_3day_mean);
df_X_ui_join8 = df_X_ui_join8.assign(ui_ck_dv3_7day=df_X_ui_join8.ui_ck_7day-df_X_ui_join8.ui_ck_7day_mean);
df_X_ui_join8 = df_X_ui_join8.assign(ui_cl_dv3_7day=df_X_ui_join8.ui_cl_7day-df_X_ui_join8.ui_cl_7day_mean);
df_X_ui_join8 = df_X_ui_join8.assign(ui_ct_dv3_7day=df_X_ui_join8.ui_ct_7day-df_X_ui_join8.ui_ct_7day_mean);
df_X_ui_join8 = df_X_ui_join8.assign(ui_by_dv3_7day=df_X_ui_join8.ui_by_7day-df_X_ui_join8.ui_by_7day_mean);
df_X_ui_join8 = df_X_ui_join8.set_index(['item_id'], append=True).reorder_levels(['user_id', 'item_id', 'item_category', 'date']);
df_X_ui_join8.drop(df_X_ui_join8.ix[:, 0:24].head(0).columns, axis=1, inplace=True);
df_X_ui_join8.drop(df_X_ui_join8.ix[:, 12:24].head(0).columns, axis=1, inplace=True);#

##build feature X
df_X = df_clean_sub.loc[:, ['user_id', 'item_id', 'item_category', 'date']];
df_X.drop_duplicates(inplace=True);

#join user# 35
df_X = df_X.join(df_X_u_join, on=['user_id', 'date'], how='left').fillna(0);

#join cate# 35
df_X = df_X.join(df_X_c_join, on=['item_category', 'date'], how='left').fillna(0);

#join item# 35+24
df_X = df_X.join(df_X_i_join, on=['item_id', 'date'], how='left').fillna(0);
df_X = df_X.join(df_X_ic_join, on=['item_id', 'item_category', 'date'], how='left').fillna(0);

#join user & cate# 12+24+24
df_X = df_X.join(df_X_uc_join, on=['user_id', 'item_category', 'date'], how='left').fillna(0);
df_X = df_X.join(df_X_uc_join1, on=['user_id', 'item_category', 'date'], how='left').fillna(0);
df_X = df_X.join(df_X_uc_join3, on=['user_id', 'item_category', 'date'], how='left').fillna(0);

#join user & item# 12+24+24+24
df_X = df_X.join(df_X_ui_join, on=['user_id', 'item_id', 'date'], how='left').fillna(0);
df_X = df_X.join(df_X_ui_join1, on=['user_id', 'item_id', 'date'], how='left').fillna(0);
df_X = df_X.join(df_X_ui_join3, on=['user_id', 'item_id', 'date'], how='left').fillna(0);
df_X = df_X.join(df_X_ui_join8, on=['user_id', 'item_id', 'item_category', 'date'], how='left').fillna(0);

##build target y
df_y = df_clean_sub.loc[:, ['user_id', 'item_id', 'item_category', 'date', 'behavior_type']];
df_y.drop_duplicates(inplace=True);
df_y = df_y.loc[df_y.behavior_type==4];
df_y.date = df_y.date.sub(np.timedelta64(1, 'D'));
df_y = df_y.rename(columns={'behavior_type': 'buy_nextday'});
df_y.buy_nextday = 1;
df_y = df_y.set_index(['user_id', 'item_id', 'item_category', 'date']);

##build feature Xy
df_Xy = df_X.join(df_y, on=['user_id', 'item_id', 'item_category', 'date'], how='left').fillna(0);

##deal feature data
#normalization
df_Xy = df_Xy.set_index(['user_id', 'item_id', 'item_category', 'date', 'buy_nextday']);
scaler = MinMaxScaler();
df_Xy_scale = scaler.fit_transform(df_Xy);
df_Xy.iloc[:] = df_Xy_scale;

##feature selection


##divide data set
df_Xy = df_Xy.reset_index(level=[3, 4]);
df_Xy_train = df_Xy.loc[(df_Xy.date>np.datetime64('2014-12-13')) & (df_Xy.date<np.datetime64('2014-12-17'))];
df_Xy_cv = df_Xy.loc[df_Xy.date==np.datetime64('2014-12-17')];
df_Xy_test = df_Xy.loc[df_Xy.date==np.datetime64('2014-12-18')];
df_Xy_train = df_Xy_train.set_index(['date'], append=True);
df_Xy_cv = df_Xy_cv.set_index(['date'], append=True);
df_Xy_test = df_Xy_test.set_index(['date'], append=True);

X_train = df_Xy_train.iloc[:, 1:];
X_cv = df_Xy_cv.iloc[:, 1:];
X_test = df_Xy_test.iloc[:, 1:];

y_train = df_Xy_train.iloc[:, 0:1];
y_cv = df_Xy_cv.iloc[:, 0:1];
y_test = df_Xy_test.iloc[:, 0:1];

##samplize train set


##design ML model
#LR
model1 = lr(class_weight='balanced');
res1 = model1.fit(X_train, y_train.values.ravel());

train_pred1 = model1.predict(X_train);
train_accuracy1 = metrics.accuracy_score(y_train.values.ravel(), train_pred1);
train_precision1 = metrics.precision_score(y_train.values.ravel(), train_pred1);
train_recall1 = metrics.recall_score(y_train.values.ravel(), train_pred1);
train_f11 = metrics.f1_score(y_train.values.ravel(), train_pred1);

cv_pred1 = model1.predict(X_cv);
cv_accuracy1 = metrics.accuracy_score(y_cv.values.ravel(), cv_pred1);
cv_precision1 = metrics.precision_score(y_cv.values.ravel(), cv_pred1);
cv_recall1 = metrics.recall_score(y_cv.values.ravel(), cv_pred1);
cv_f11 = metrics.f1_score(y_cv.values.ravel(), cv_pred1);

test_pred1 = model1.predict(X_test);

#svm
n_estimators = 10;
model2 = BaggingClassifier(LinearSVC(class_weight='balanced'), max_samples=1.0 / n_estimators, n_estimators=n_estimators, verbose=20);
res2 = model2.fit(X_train, y_train.values.ravel());

train_pred2 = model2.predict(X_train);
train_accuracy2 = metrics.accuracy_score(y_train.values.ravel(), train_pred2);
train_precision2 = metrics.precision_score(y_train.values.ravel(), train_pred2);
train_recall2 = metrics.recall_score(y_train.values.ravel(), train_pred2);
train_f12 = metrics.f1_score(y_train.values.ravel(), train_pred2);

cv_pred2 = model2.predict(X_cv);
cv_accuracy2 = metrics.accuracy_score(y_cv.values.ravel(), cv_pred2);
cv_precision2 = metrics.precision_score(y_cv.values.ravel(), cv_pred2);
cv_recall2 = metrics.recall_score(y_cv.values.ravel(), cv_pred2);
cv_f12 = metrics.f1_score(y_cv.values.ravel(), cv_pred2);

test_pred2 = model2.predict(X_test);



