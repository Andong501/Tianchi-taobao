天池数据挖掘比赛-淘宝用户购物推荐预测
===============
Tianchi data mining competion-Taobao parchase prediction
---------------

[比赛网址](https://tianchi.aliyun.com/getStart/introduction.htm?raceId=231522)
语言版本：python 2.7

## 目录及说明
### 1.tianchi_taobao_analysis  天池数据挖掘比赛-淘宝用户购物预测数据分析部分
####     * introduction.ipynb  数据分析报告 [快速查看](https://nbviewer.jupyter.org/github/Andong501/Tianchi-taobao/blob/master/introduction.ipynb)
####     * Tianchi competition-data analysis.py  数据分析程序
### 2.tianchi_taobao_prediction_v1.0  天池数据挖掘比赛-淘宝用户购物预测部分v1.0
####     * Tianchi competition-Taobao purchase prediction1.py  购物预测模型训练及预测v1.0
####     * tianchi_mobile_recommendation_predict1.csv  预测结果v1.0
### 3.tianchi_taobao_prediction_v2.0  天池数据挖掘比赛-淘宝用户购物预测部分v2.0
####     * Feature instruction  新增特征构造说明文档
####     * Tianchi competition-Taobao purchase prediction2.py  购物预测模型训练及预测v2.0
####     * tianchi_mobile_recommendation_predict2.csv  预测结果v2.0
### 4.tianchi_taobao_prediction_v3.0  天池数据挖掘比赛-淘宝用户购物预测部分v3.0
####     * Feature instruction plus  新增特征构造说明文档新增规则部分
####     * Tianchi competition-Taobao purchase prediction3.py  购物预测模型训练及预测v3.0
####     * tianchi_mobile_recommendation_predict3.csv  预测结果v3.0

## 详细说明
### 第一次模型训练及预测
#### 训练集: 90,000 X 4
#### cv: 30,000 X 4
#### 线下测试集: 30,000 X 4
#### 线上测试集: 30,000 X 4

#### 模型1: LR(class_weight='balanced')
#### f1_train:0.051
#### f1_cv:0.058
#### f1_local_test:0.062
#### f1_test: 0.058
#### rank: 10/5410

#### 模型2：svm+bagging(class_weight='balanced')
#### f1_test: 0.067
#### rank: 12/5410

### 第二次模型训练及预测
#### 优化理由：经过第一次模型预测后发现，由于特征量过少（4个），存在一定的欠拟合情况（训练集效果劣于cv、测试集），因此决定以增加特征的方式进行优化。
#### 训练集: 90,000 X 273
#### cv: 28,000 X 273
#### 线上测试集: 29,000 X 273

#### 模型：svm+bagging(class_weight='balanced')
#### f1_train:0.078
#### f1_cv:0.064
#### f1_test:0.0703
#### rank: 27/5740

### 第三次模型训练及预测
####优化理由：使用比赛中表现最佳的GBDT模型，通过10次不同的训练集采样得到10个GBDT模型，将10个GBDT模型进行bagging voting；同时添加了基于业务的附加规则；此外由于GBDT的训练用时较长，此版本将模型训练部分迁移至Spark ML框架进行，使训练速度大幅提升。
#### 训练集：16,800 X 273
#### cv:28,000 X 273
#### 线上测试集: 29，000 X273

#### 模型：GBDT+bagging+rule
#### f1_train:0.086
#### f1_cv:0.082
#### f1_test:0.088
#### rank:20/5400

### To be continued...
