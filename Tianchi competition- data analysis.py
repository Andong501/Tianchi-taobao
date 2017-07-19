#Tianchi competition: data analysis

import matplotlib as mlt
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
import numpy as np 
import pandas as pd 
import datetime
from __future__ import division #精确除法

#import csv
df1 = pd.read_csv('tianchi_fresh_comp_train_user.csv');
df2 = pd.read_csv('tianchi_fresh_comp_train_item.csv');

#basic analusis
#total users
total_user = len(df1['user_id'].value_counts().index);
#total items
total_item = len(df1['item_id'].value_counts().index);
#predict items
pre_item = len(df2['item_id'].value_counts().index);

#purchase distribution by date
#extract part of cases
df1_par = df1.loc[:, ['user_id', 'item_id', 'time', 'behavior_type', 'item_category', 'user_geohash']];
df1_par.time = pd.to_datetime(df1_par.time).values.astype('datetime64[D]');
df1_par.set_index('time', drop=False, append=False, inplace=True, verify_integrity=False);
df1_par = df1_par.sort_index();
g1 = sns.countplot(x='time', data=df1_par);
g1.set_xticklabels(g1.get_xticklabels(), rotation=90);

#sale distribution by product class
g2 = sns.countplot('item_category', data=df1_par);
g2.set_xticklabels(g2.get_xticklabels(), rotation=90);

#sale distribution by geo
g3 = sns.countplot('user_geohash', data=df1_par.loc[df1_par.user_geohash!=None]);
g3.set_xticklabels(get_xticklabels(), rotation=90);

#Top 100 sales by production class
df1_par.item_category.value_counts().sort_values(ascending=False).iloc[0:100];

#Top 100 sales by geo
df1_par.user_geohash.value_counts().sort_values(ascending=False).iloc[0:100];

#the buying chance by whether click/collete/cart/buy on the days before 12-18
#df1_da = df1_DataAnalysis
#df1_da1 = df1_DataAnalysis1
df1_da = df1.loc[:, ['user_id', 'item_id', 'behavior_type', 'time']];
df1_da.time = pd.to_datetime(df1_da.time);
df1_da.time = df1_da.time.values.astype('datetime64[D]'); #fentch date only
df1_da1 = df1_da;
final_date = np.datetime64('2014-12-18');
df1_da1.time = df1_da1.time.sub(final_date).mul(-1);

s_click = (df1_da1.behavior_type==1);
s_collec = (df1_da1.behavior_type==2);
s_cart = (df1_da1.behavior_type==3);
s_buy = (df1_da1.behavior_type==4);
d1 = {'click':s_click, 'collection':s_collec, 'cart':s_cart, 'buy':s_buy};
df1_add1 = pd.DataFrame(d1);
df1_tol1 = pd.concat([df1_da1, df1_add1], axis=1);
df1_tol1.set_index(['user_id', 'item_id'], drop=True, append=False, inplace=True, verify_integrity=False);

s_buy1218 = ((df1_da1.time==pd.tslib.Timedelta('0 days 00:00:00')) & (df1_da1.behavior_type==4));
d2 = {'buy1218':s_buy1218};
df1_add2 = pd.DataFrame(d2);
df1_tol2 = pd.concat([df1_da1, df1_add2], axis=1);
df1_tol2.drop(['time', 'behavior_type'], axis=1, inplace=True);
df1_tol2 = df1_tol2.loc[df1_tol2.buy1218==True];
df1_tol2.set_index(['user_id', 'item_id'], drop=True, append=False, inplace=True, verify_integrity=False);

df1_tol = df1_tol1.join(df1_tol2).fillna(False); #review join/merge/concat!

#by click
chance_click = [0]*31;
for i in range(0,31):
	hit = df1_tol.buy1218.loc[(df1_tol.time==pd.tslib.Timedelta(str(i)+' days 00:00:00')) & (df1_tol.click==True) & (df1_tol.buy1218==True)].value_counts(dropna=False).sum();
	total = df1_tol.buy1218.loc[(df1_tol.time==pd.tslib.Timedelta(str(i)+' days 00:00:00')) & (df1_tol.click==True)].value_counts(dropna=False).sum();
	chance_click[i] = hit/total;
dic_click = {'chance_click':chance_click, 'day':range(0, 31)};
df1_click = pd.DataFrame(dic_click);
sns.barplot(x='day', y='chance_click', data=df1_click);

#by collect
chance_collec = [0]*31;
for i in range(0,31):
	hit = df1_tol.buy1218.loc[(df1_tol.time==pd.tslib.Timedelta(str(i)+' days 00:00:00')) & (df1_tol.collection==True) & (df1_tol.buy1218==True)].value_counts(dropna=False).sum();
	total = df1_tol.buy1218.loc[(df1_tol.time==pd.tslib.Timedelta(str(i)+' days 00:00:00')) & (df1_tol.collection==True)].value_counts(dropna=False).sum();
	chance_collec[i] = hit/total;
dic_collec = {'chance_collec':chance_collec, 'day':range(0, 31)};
df1_collec = pd.DataFrame(dic_collec);
sns.barplot(x='day', y='chance_collec', data=df1_collec);

#by cart
chance_cart = [0]*31;
for i in range(0,31):
	hit = df1_tol.buy1218.loc[(df1_tol.time==pd.tslib.Timedelta(str(i)+' days 00:00:00')) & (df1_tol.cart==True) & (df1_tol.buy1218==True)].value_counts(dropna=False).sum();
	total = df1_tol.buy1218.loc[(df1_tol.time==pd.tslib.Timedelta(str(i)+' days 00:00:00')) & (df1_tol.cart==True)].value_counts(dropna=False).sum();
	chance_cart[i] = hit/total;
dic_cart = {'chance_cart':chance_cart, 'day':range(0, 31)};
df1_cart = pd.DataFrame(dic_cart);
sns.barplot(x='day', y='chance_cart', data=df1_cart);

#by buy
chance_buy = [0]*31;
for i in range(0,31):
	hit = df1_tol.buy1218.loc[(df1_tol.time==pd.tslib.Timedelta(str(i)+' days 00:00:00')) & (df1_tol.buy==True) & (df1_tol.buy1218==True)].value_counts(dropna=False).sum();
	total = df1_tol.buy1218.loc[(df1_tol.time==pd.tslib.Timedelta(str(i)+' days 00:00:00')) & (df1_tol.buy==True)].value_counts(dropna=False).sum();
	chance_buy[i] = hit/total;
dic_buy = {'chance_buy':chance_buy, 'day':range(0, 31)};
df1_buy = pd.DataFrame(dic_buy);
g4 = sns.barplot(x='day', y='chance_buy', data=df1_buy);
g4.set_yscale('log')

#the buying chance by the number of click/collete/cart/buy the item one day before
#df1_da2 = da1_DataAnalysis2
df1_da2 = df1_da;
df1_add3 = pd.DataFrame(d1);
df1_tol3 = pd.concat([df1_da2, df1_add3], axis=1);
df1_tol3.drop(['behavior_type'], axis=1, inplace=True);
df1_tol3_gp = df1_tol3.groupby(['user_id', 'item_id', 'time']).sum(); #count and delete duplication

df1_tol4 = df1_tol3.drop(['click', 'collection', 'cart'], axis=1, inplace=False);
one_day = pd.Series(np.array([np.timedelta64(1,'D')]*len(df1_tol4)));
df1_tol4.time = df1_tol4.time - one_day;
df1_tol4_gp = df1_tol4.groupby(['user_id', 'item_id', 'time']).sum();
df1_tol4_gp.buy = (df1_tol4_gp.buy!=0);

df1_tol_gp = df1_tol3_gp.join(df1_tol4_gp, rsuffix='_next_day'); #know the meaning of buy_next_day=True/False/NaN
df1_tol_gp.loc[:, 'buy_next_day'].fillna(False, inplace=True);

#by click times
sns.barplot(x='click', y='buy_next_day', data=df1_tol_gp);
#by collect
sns.barplot(x='collection', y='buy_next_day', data=df1_tol_gp);
#by cart
sns.barplot(x='cart', y='buy_next_day', data=df1_tol_gp);
#by buy
sns.barplot(x='buy', y='buy_next_day', data=df1_tol_gp);
