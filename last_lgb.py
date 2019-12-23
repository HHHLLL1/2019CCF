# -*- coding: utf-8 -*-
"""
Created on Sat Nov 23 00:35:36 2019

@author: Lenovo
"""





import numpy as np
import pandas as pd
import catboost as cbt
import lightgbm as lgb
from sklearn.metrics import accuracy_score, roc_auc_score, log_loss
import gc
import math
import time
from tqdm import tqdm
import datetime
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import warnings
import os
from sklearn.utils import shuffle

warnings.filterwarnings('ignore')
pd.options.display.max_columns = None
pd.options.display.max_rows = None



'''
读取数据
train.csv为初赛和复赛的训练数据concat而成
'''
train = pd.read_csv('./data/train.csv')
test = pd.read_csv('./data/second_round_testing_data.csv')
submit = pd.read_csv('./data/submit_example2.csv')


start_index = 1
end_index = 11

#处理label
data = train.append(test).reset_index(drop=True)
dit = {'Excellent': 0, 'Good': 1, 'Pass': 2, 'Fail': 3}
data['label'] = data['Quality_label'].map(dit)

#填充Parameter9，用前一个值填充
data['Parameter9'] = data['Parameter9'].fillna(method='ffill')
data['Parameter9'] = data['Parameter9'].fillna(method='bfill')


#处理原始特征，取对数
for i in range(1, 11):
    data[f'Parameter{i}'] = np.log10(data[f'Parameter{i}'])
    
for i in range(1, 11):
    data[f'Attribute{i}'] = np.log10(data[f'Attribute{i}'])


#####################################################################################

'''
提取人工特征，用于构造多项式特征
'''

feature_name1 = []

data['10_9'] = data['Parameter10'].astype('str') + '_' + data['Parameter9'].astype('str')
col = '10_9'
d = pd.DataFrame()
d[f'{col}_count'] = data.groupby([col]).size()
data = pd.merge(data, d, on=col, how='left')
feature_name1.append(f'{col}_count')

data['10_8'] = data['Parameter10'].astype('str') + '_' + data['Parameter8'].astype('str')
col = '10_8'
d = pd.DataFrame()
d[f'{col}_count'] = data.groupby([col]).size()
data = pd.merge(data, d, on=col, how='left')
feature_name1.append(f'{col}_count')


data['7_8_10'] = data['Parameter7'].astype('str') + '_' + data['Parameter8'].astype('str') + '_' + data['Parameter10'].astype('str')
col = '7_8_10'
d = pd.DataFrame()
d[f'{col}_count'] = data.groupby([col]).size()
data = pd.merge(data, d, on=col, how='left')
feature_name1.append(f'{col}_count')


data['10*9'] = data['Parameter10'] * data['Parameter9']
data['10*5'] = data['Parameter10'] * data['Parameter5']
data['10*8'] = data['Parameter10'] * data['Parameter8']
data['10*7'] = data['Parameter10'] * data['Parameter7']
data['10*6'] = data['Parameter10'] * data['Parameter6']
data['10*10'] = data['Parameter10'] * data['Parameter10']
data['9*7'] = data['Parameter9'] * data['Parameter7']
data['9*6'] = data['Parameter9'] * data['Parameter6']
data['9*5'] = data['Parameter9'] * data['Parameter5']
data['8*7'] = data['Parameter8'] * data['Parameter7']
data['7*6'] = data['Parameter7'] * data['Parameter6']
data['6*7'] = data['Parameter6'] * data['Parameter7']
data['6*5'] = data['Parameter6'] * data['Parameter5']


data['10/9'] = data['Parameter10'] / data['Parameter9']
data['10/8'] = data['Parameter10'] / data['Parameter8']
data['10/7'] = data['Parameter10'] / data['Parameter7']
data['10/6'] = data['Parameter10'] / data['Parameter6']
data['10/5'] = data['Parameter10'] / data['Parameter5']
data['9/8'] = data['Parameter9'] / data['Parameter8']
data['8/6'] = data['Parameter8'] / data['Parameter6']
data['8/5'] = data['Parameter8'] / data['Parameter5']
data['7/10'] = data['Parameter7'] / data['Parameter10']
data['7/8'] = data['Parameter7'] / data['Parameter8']
data['5/7'] = data['Parameter5'] / data['Parameter7']

data['10*8/9'] = data['10*8'] / data['Parameter9']
data['5/6'] = data['Parameter5'] / data['Parameter6']


feature_name1.append('10*8/9')
feature_name1.append('9*7')#################
feature_name1.append('9*5')################
feature_name1.append('6*5')
feature_name1.append('10/9')############
feature_name1.append('10/8')
feature_name1.append('8/5')
feature_name1.append('7/10')
feature_name1.append('5/6')


#data['10*8'] = data['Parameter10'] * data['Parameter8']
#data['10*8/9'] = data['10*8'] / data['Parameter9']
#feature_name1.append('10*8/9')

#倒数特征
data['10dao'] = 1 / data['Parameter10']    
feature_name1.append('10dao')

#类别特征提取count特征
cat_feature = ['Parameter5','Parameter6','Parameter7','Parameter8','Parameter9']
for col in cat_feature:
    temp=pd.DataFrame()
    temp[f'{col}_count'] = data.groupby([col]).size()
    data = pd.merge(data, temp, on=col, how='left')
    feature_name1.append(f'{col}_count')






feature_name = ['Parameter{0}'.format(i) for i in range(start_index, end_index)]
#feature_name.remove('Parameter1')
#feature_name.remove('Parameter4')
tr_index = ~data['label'].isnull()
X_train_org = data[tr_index][feature_name].reset_index(drop=True)
x_train_ath = data[tr_index][feature_name1].reset_index(drop=True)
y = data[tr_index]['label'].reset_index(drop=True).astype(int)
X_test_org = data[~tr_index][feature_name].reset_index(drop=True)
x_test_ath = data[~tr_index][feature_name1].reset_index(drop=True)


###################################################################################

'''
处理原始特征，用于特征选择
'''

#根据label的类别进行抽取，对数据进行去中心化
X_train_dis = pd.DataFrame()
X_test_dis = pd.DataFrame()
#print(y)
classdf = pd.DataFrame(y)
for classindex in range(4):
    class_row_index_list = classdf[(classdf['label'] == classindex)].index.tolist()

    df_train_temp = X_train_org.iloc[class_row_index_list]

    print('class{0} is len is {1}'.format(classindex, len(df_train_temp)))

    for Parameter in range(start_index, end_index):
        Parameter_std = df_train_temp['Parameter{0}'.format(Parameter)].std()
        Parameter_mean = df_train_temp['Parameter{0}'.format(Parameter)].mean()
        X_train_dis['Parameter{0}class{1}dis_mean'.format(Parameter, classindex)] = X_train_org['Parameter{0}'.format(
            Parameter)] - Parameter_mean
        X_test_dis['Parameter{0}class{1}dis_mean'.format(Parameter, classindex)] = X_test_org['Parameter{0}'.format(
            Parameter)] - Parameter_mean

    
    
    
############################################################################################    
    

#使用sklearn进行构建多项式特征
from sklearn.preprocessing import PolynomialFeatures
from sklearn.feature_selection import SelectPercentile, VarianceThreshold

def make_feature(df_temp, degree_level):
    poly = PolynomialFeatures(degree=degree_level, include_bias=False, interaction_only=False)
    X_ploly = poly.fit_transform(df_temp)
    df_temp = pd.DataFrame(X_ploly, columns=poly.get_feature_names())
    return df_temp

X_train_org_ploly3 = pd.DataFrame()
X_test_org_ploly3 = pd.DataFrame()

#标准化原始数据
for Parameter in range(start_index, end_index):
    Parameter_std = X_train_org['Parameter{0}'.format(Parameter)].std()
    Parameter_mean = X_train_org['Parameter{0}'.format(Parameter)].mean()
    X_train_org_ploly3['Parameter{0}'.format(Parameter)] = (X_train_org['Parameter{0}'.format(
        Parameter)] - Parameter_mean) / Parameter_std
    X_test_org_ploly3['Parameter{0}'.format(Parameter)] = (X_test_org['Parameter{0}'.format(
        Parameter)] - Parameter_mean) / Parameter_std

#组合原始特征和人工特征
X_train_org_ploly3 = pd.concat([X_train_org_ploly3, x_train_ath], axis=1)
X_test_org_ploly3 = pd.concat([X_test_org_ploly3, x_test_ath], axis=1)
del x_train_ath
del x_test_ath
gc.collect()

#组合训练集合测试集，一起构造多项式特征
d = pd.concat((X_train_org_ploly3, X_test_org_ploly3), axis=0, ignore_index=True)
d = make_feature(d, 3)

#分离训练集、测试集
X_train_ploly_3 = d.iloc[:len(train)]
X_test_ploly_3 = d.iloc[len(train):]
del d
gc.collect()


#############################################################################################

'''
多项式特征过多，
使用单变量特征选择筛选特征，
减少特征数量。
'''


select = SelectPercentile(percentile=15)
select.fit(X_train_ploly_3, y)

mask = select.get_support()
print(mask)

feature_used = []
for index in range(len(mask)):
    if mask[index]:
        feature_used.append(list(X_train_ploly_3.columns.values)[index])

X_train_ploly_3 = X_train_ploly_3[feature_used]
X_test_ploly_3 = X_test_ploly_3[feature_used].reset_index()
X_test_ploly_3.drop('index', axis=1, inplace=True)
    
#X_train_ploly_3 = make_feature(X_train_org_ploly3, 4)
#select = SelectPercentile(percentile=100)  # 选择特征重要度排在前百分之35的特征,默认计算函数是f_classif(只适用于分类问题)
#select.fit(X_train_ploly_3, y)
#
#mask = select.get_support()
#print(mask)
#
#feature_used = []
#for index in range(len(mask)):
#    if mask[index]:
#        feature_used.append(list(X_train_ploly_3.columns.values)[index])
#
#X_train_ploly_3 = X_train_ploly_3[feature_used]
#X_test_ploly_3 = make_feature(X_test_org_ploly3, 3)
#X_test_ploly_3 = X_test_ploly_3[feature_used]

print(X_train_ploly_3.shape, X_test_ploly_3.shape)




#####################################################################################

'''
根据数据预分析，
对每一列特征聚类，
聚类类别数由手肘法确定。
(但是效果似乎不太好，此类特征在最后的特征筛选中基本被去掉了)
'''


clusterdic = {1: 5, 2: 4, 3: 4, 4: 6, 5: 6, 6: 6, 7: 3, 8: 4, 9: 2, 10: 5}

from sklearn.cluster import KMeans


def getclusterfeature(df_train, df_predict1, df_predict2):
    df_temp1 = pd.DataFrame()
    df_temp2 = pd.DataFrame()
    for i in range(start_index, end_index):
        colindex = i
        df_features = df_train[['Parameter{0}'.format(colindex)]]

        estimator = KMeans(n_clusters=clusterdic[colindex])  # 构造聚类器
        estimator.fit(df_features)
        predictclass1 = estimator.predict(df_predict1[['Parameter{0}'.format(colindex)]])
        df_temp1['Parametercluter{0}'.format(colindex)] = predictclass1.tolist()
        predictclass2 = estimator.predict(df_predict2[['Parameter{0}'.format(colindex)]])
        df_temp2['Parametercluter{0}'.format(colindex)] = predictclass2.tolist()

    return df_temp1, df_temp2

d = data[feature_name + feature_name1]
X_train_cluster, X_test_cluster = getclusterfeature(d, X_train_org, X_test_org)
print(X_train_cluster.shape, X_test_cluster.shape)


########################################################################################

'''
组合之前的特征，
做单变量特征选择，
进行B类特征的预测。
'''

percent = 50
#X_train = pd.concat([X_train_dis, X_train_ploly_3, X_train_cluster, x_train_ath], axis=1)
#X_test = pd.concat([X_test_dis, X_test_ploly_3, X_test_cluster, x_test_ath], axis=1)
X_train = pd.concat([X_train_dis, X_train_ploly_3, X_train_cluster], axis=1)
X_test = pd.concat([X_test_dis, X_test_ploly_3, X_test_cluster], axis=1)


#list_file = open('percent.txt', 'a')
#print('percent is {0}'.format(percent))

select = SelectPercentile(percentile=percent)
select.fit(X_train, y)

mask = select.get_support()
print(mask)

feature_used = []
for index in range(len(mask)):
    if mask[index]:
        feature_used.append(list(X_train.columns.values)[index])
print('最终特征选择')
print(feature_used)
#list_file.write(','.join(feature_used) + '\n')
X_train = X_train[feature_used]
X_test = X_test[feature_used]
print('最终特征维度')
print(X_train.shape, X_test.shape)



#################################################################################

'''
B类特征预测函数
'''

def get_predict_w(model, data, label='label', feature=[], cate_feature=[], random_state=2018, n_splits=5,
                  model_type='lgb'):
    if 'sample_weight' not in data.keys():
        data['sample_weight'] = 1
    model.random_state = random_state
    predict_label = 'predict_' + label
    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    data[predict_label] = 0
    test_index = (data[label].isnull()) | (data[label] == -1)
    train_data = data[~test_index].reset_index(drop=True)
#    train_data[features] = (train_data[features] - train_data[features].min())/(train_data[features].max() - train_data[features].min())
    test_data = data[test_index]
#    test_data[features] = (test_data[features] - test_data[features].min())/(test_data[features].max() - test_data[features].min())


    for train_idx, val_idx in kfold.split(train_data):
        model.random_state = model.random_state + 1

        train_x = train_data.loc[train_idx][feature]
        train_y = train_data.loc[train_idx][label]

        test_x = train_data.loc[val_idx][feature]
        test_y = train_data.loc[val_idx][label]
        if model_type == 'lgb':
            try:
                model.fit(train_x, train_y, eval_set=[(test_x, test_y)], early_stopping_rounds=100,
                          eval_metric='mae',
                          categorical_feature=cate_feature,
                          sample_weight=train_data.loc[train_idx]['sample_weight'],
                          verbose=300
                          )
#                model.fit(train_x, train_y)
            except:
                model.fit(train_x, train_y, eval_set=[(test_x, test_y)], early_stopping_rounds=100,
                          eval_metric='mae',
                          # categorical_feature=cate_feature,
                          sample_weight=train_data.loc[train_idx]['sample_weight'],
                          verbose=300)
#                model.fit(train_x, train_y)
            train_data.loc[val_idx, predict_label] = model.predict(test_x, num_iteration=model.best_iteration_)
            if len(test_data) != 0:
                test_data[predict_label] = test_data[predict_label] + model.predict(test_data[feature], num_iteration=model.best_iteration_)

        elif model_type == 'cbt':
#            model.fit(train_x, train_y, eval_set=[(test_x, test_y)], early_stopping_rounds=100,
#                      eval_metric='mae',
#                      cat_features=cate_feature,
#                      sample_weight=train_data.loc[train_idx]['sample_weight'],
#                      verbose=100)
            model.fit(train_x, train_y, eval_set=[(test_x, test_y)], early_stopping_rounds=100,
                      sample_weight=train_data.loc[train_idx]['sample_weight'],
                      verbose=300, use_best_model=True)

            train_data.loc[val_idx, predict_label] = model.predict(test_x)
            if len(test_data) != 0:
                test_data[predict_label] = test_data[predict_label] + model.predict(test_data[feature])
                
    test_data[predict_label] = test_data[predict_label] / n_splits
    return pd.concat([train_data, test_data], sort=True, ignore_index=True), predict_label


#预测测试集的Attribute特征

lgb_attr_model = lgb.LGBMRegressor(
    boosting_type="gbdt", num_leaves=31, reg_alpha=10, reg_lambda=5,
    max_depth=7, n_estimators=10000,
    subsample=0.7, colsample_bytree=0.4, subsample_freq=2, min_child_samples=10,
    learning_rate=0.1, random_state=2019
)

cbt_attr_model = cbt.CatBoostRegressor(
        num_leaves=31, 
#        reg_lambda=5,
        max_depth=7, 
#        n_estimators=10000,
        n_estimators=2000,
#        subsample=0.7, 
#        min_child_samples=10,
        learning_rate=0.1, 
        random_state=2,
        eval_metric='MAE',
        task_type='GPU'
        )

##组合需要的数据，方便训练
#gpr  = GaussianProcessRegressor()
tr_len = len(X_train)
data1 = pd.concat((X_train, X_test), axis=0, ignore_index=True)
features = list(data1.columns)

ffff = ['Attribute1', 'Attribute10', 'Attribute2', 'Attribute3', 'Attribute4',
       'Attribute5', 'Attribute6', 'Attribute7', 'Attribute8', 'Attribute9', 'Parameter5',
       'Parameter6']
data1 = pd.concat((data[ffff+['label']], data1), axis=1)

#建立训练特征列表
features = features + ['Parameter5','Parameter6']

'''
预测B类4-6特征，
使用cbt进行预测，
增大训练次数和学习率，
提高精度加快训练速度，
选取score最好模型预测.
'''
attr_feat = ['Attribute{0}'.format(i) for i in range(4, 7)]
for i in attr_feat:
    data1, predict_label = get_predict_w(cbt_attr_model, data1, label=i,
                                        feature=features, random_state=2019, n_splits=10, model_type='cbt')
    print(predict_label, 'done!!')


##features = para_feat
#for i in attr_feat:
#    data, predict_label = get_predict_w(cbt_attr_model, data, label=i,
#                                        feature=features, random_state=2, n_splits=5, model_type='cbt')
#    data, predict_label = get_predict_w(lgb_attr_model, data, label=i,
#                                        feature=features, random_state=2, n_splits=5, model_type='lgb')
#    print(predict_label, 'done!!\n')
#
#
#for i in attr_feat:
#    data[f'predict_{i}'] = (data['lgb_predict_' + i] + data['cbt_predict_' + i])/2
    
    
# 该方案共获得10个属性特征。
pred_attr_feat = ['predict_Attribute{0}'.format(i) for i in range(4, 7)]


'''
预测出的B类4-6特征和之前的特征合并，
作为最后的特征进行结果预测
'''
features.extend(pred_attr_feat)


col_remove = ['Attribute1', 'Attribute10', 'Attribute2', 'Attribute3', 'Attribute4',
              'Attribute5', 'Attribute6', 'Attribute7', 'Attribute8', 'Attribute9']
features = [i for i in features if i not in col_remove]
print(features)    


    
#划分数据    
print(X_train.shape, X_test.shape)
tr_index = ~data1['label'].isnull()
X_train = data1[tr_index][features].reset_index(drop=True)
y = data1[tr_index]['label'].reset_index(drop=True).astype(int)
X_test = data1[~tr_index][features].reset_index(drop=True)



###################################################################################




'''
开始训练，
任意固定一个种子，
skf十折交叉验证，
使用lgb
'''

oof = np.zeros((X_train.shape[0],4))
prediction = np.zeros((X_test.shape[0],4))
num_model_seed = 1
seeds = [2, 19970412, 2019 * 2 + 1024, 4096, 2048, 1024]
#seeds = range(10)
score = []





cbt_model = cbt.CatBoostClassifier(iterations=4000,learning_rate=0.01,verbose=300,
                                   max_depth=7,#eval_metric='MAE',
                                   early_stopping_rounds=100,
                                   task_type='GPU',#custom_metric='Accuracy',
                                   loss_function='MultiClass')


lgb_model = lgb.LGBMClassifier( boosting_type="gbdt", num_leaves=23, reg_alpha=10, reg_lambda=5,
                                max_depth=5, n_estimators=4000, objective='multiclass',
                                subsample=0.7, colsample_bytree=0.7, subsample_freq=1, min_child_samples=5,
                                learning_rate=0.05, random_state=42,
                            )




for model_seed in range(num_model_seed):
    print('\n',model_seed + 1)
    oof_cat = np.zeros((X_train.shape[0],4))
    prediction_cat=np.zeros((X_test.shape[0],4))
    
    num_skf = 10
    skf = StratifiedKFold(n_splits=num_skf, random_state=seeds[model_seed], shuffle=True)
    for index, (train_index, test_index) in enumerate(skf.split(X_train, y)):
        print(index)
        train_x, test_x, train_y, test_y = X_train.iloc[train_index], X_train.iloc[test_index], y.iloc[train_index], y.iloc[test_index]

#        cbt_model.fit(train_x, train_y, eval_set=(test_x, test_y), early_stopping_rounds=100, use_best_model=True)
#        oof_cat[test_index] += cbt_model.predict_proba(test_x)
#        prediction_cat += cbt_model.predict_proba(X_test)/num_skf
#        gc.collect()

        eval_set = [(test_x, test_y)]
        lgb_model.fit(train_x, train_y, eval_set=eval_set,early_stopping_rounds=100,verbose=100)    
        oof_cat[test_index] += lgb_model.predict_proba(test_x, num_iteration=lgb_model.best_iteration_)
        prediction += lgb_model.predict_proba(X_test, num_iteration=lgb_model.best_iteration_)/num_skf
        gc.collect()

                
        
    oof += oof_cat / num_model_seed
    prediction += prediction_cat / num_model_seed
        
    print('logloss',log_loss(pd.get_dummies(y).values, oof_cat))
    print('ac',accuracy_score(y, np.argmax(oof_cat,axis=1)))
    print('mae',1/(1 + np.sum(np.absolute(np.eye(4)[y] - oof_cat))/480))
     
print('logloss',log_loss(pd.get_dummies(y).values, oof))
print('ac',accuracy_score(y, np.argmax(oof,axis=1)))
print('mae',1/(1 + np.sum(np.absolute(np.eye(4)[y] - oof))/480))

#feature_importance = pd.DataFrame(cbt_model.feature_importances_, index=cbt_model.feature_names_)


#######################################################################################3


#线下测试
bin_label = ['bin1', 'bin2', 'bin3', 'bin4']
oof_pred = pd.DataFrame(oof_cat, columns=bin_label)
oof_y = pd.get_dummies(y, columns='label', prefix='pred')
pred_label = oof_y.columns.tolist()


data = pd.concat([data[data.label.notnull()], oof_pred], axis=1)
data = pd.concat([data[data.label.notnull()], oof_y], axis=1)
data = data.sample(6000, random_state=2)
data.reset_index(inplace=True)



def gen_sample(data, group_values, seed=0):
    group_values = shuffle(group_values, random_state=seed)
    data['Group'] = seed*1000 + group_values
    return data

group_values = test.Group.values.copy()
data_2 = data.copy()
for i in range(1, 50):
    data_2 = pd.concat([
            gen_sample(data[~data.label.isnull()], group_values, i), data_2], ignore_index=True)
#    print(i, data_2.shape)
    
data_3 = data_2.groupby(['Group'])[pred_label + bin_label].mean().reset_index()

xck = 1/(1+10 * abs(data_3[data_3.Group>=120][pred_label].values - data_3[data_3.Group>=120][bin_label].values).mean())

print(f'线下成绩为{xck}')
gc.collect()
    

########################################################################################


sub = test[['Group']]
prob_cols = [i for i in submit.columns if i not in ['Group']]
for i, f in enumerate(prob_cols):
    sub[f] = prediction[:, i]
for i in prob_cols:
    sub[i] = sub.groupby('Group')[i].transform('mean')
sub = sub.drop_duplicates()
#sub.to_csv('./last/last_lgb.csv', index=False)






