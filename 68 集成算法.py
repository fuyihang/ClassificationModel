#-*- coding: utf-8 -*-

########  本文件实现Bagging集成元算法，包括
# Part1、分类：BaggingClassifier
# Part2、投票：VotingClassifier
# Part3、超参优化：VotingClassifier/GridSearchCV

# Part4、回归：BaggingRegressor
# Part5、投票：VotingRegressor
###################################

import pandas as pd
import numpy as np

from common import displayClassifierMetrics
from common import displayROCurve

# Bagging
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import VotingClassifier

from sklearn.ensemble import BaggingRegressor
from sklearn.ensemble import VotingRegressor

# 基学习器
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm        import SVC

# 超参优化
from sklearn.model_selection import GridSearchCV


######################################################################
########  Part1、通用的Bagging优化算法:BaggingClassifier
######################################################################

# 1、读入数据
filename = '分类预测.xls'
sheet = '银行贷款'
df = pd.read_excel(filename, sheet)
# print(df.columns.tolist())

# 2、数据处理
# 1)特征标识
intCols = ['年龄']
catCols = ['收入级别', '信用卡数', '学历', '车贷数量']

target = '违约'
y = df[target]
poslabel = '是'

# 2）类别自变量数字化
from sklearn.preprocessing import OrdinalEncoder

enc = OrdinalEncoder(dtype='int')
X_ = enc.fit_transform(df[catCols])

dfCats = pd.DataFrame(X_, columns=catCols)

for i, cats in enumerate(enc.categories_):
    print('特征:{}-{}'.format(catCols[i],cats))

# 3）合并
X = pd.concat([dfCats, df[intCols]], axis=1)
cols = X.columns.tolist()


# 3、建立模型
# 假定基模型采用决策树
mdl = DecisionTreeClassifier(
            criterion = 'gini',
            min_samples_split=100,
            min_samples_leaf=50,
            max_depth=4)
# 指定50个基模型，及其它参数
clf = BaggingClassifier(
            base_estimator=mdl, 
            n_estimators=10,    #基学习器个数
            max_samples=0.8,    #取样本的数，0.8=80%
            # max_features=1.0,   #取特征的数, 1.0=100%
            # bootstrap=True,     #是否对样本有放回抽样
            # bootstrap_features=False, #是否对特征有放回抽样
            oob_score=True,    #是否评估袋外数据集的得分
            # warm_start=False,   #是否重用原来的评估器
            # n_jobs=None,        #并行数量
            # verbose=0,          #控制复杂度参数
            random_state=10)    #随机种子
clf.fit(X, y)

# 查看元算法的其它信息，熟悉如何使用相关属性
print('基学习器的个数：', clf.n_estimators)
print('使用的特征个数：',clf.n_features_)
print('类别个数：', clf.n_classes_)
print('标签取值：', clf.classes_)
print('元算法的袋外评分：', clf.oob_score_)
# 查看基学习器，使用的特征和样本
for i in range(clf.n_estimators):
    # mdl = clf.estimators_[i]
    idxs = clf.estimators_features_[i]  #特征序号
    idxs.sort()

    estFeatures = []
    for idx in idxs:
        estFeatures.append(cols[idx])
    print('第{0}个基类使用的特征:{1}'.format(i, estFeatures) )

# 4、评估模型
y_pred = clf.predict(X)
displayClassifierMetrics(y, y_pred, clf.classes_, poslabel)

y_prob = clf.predict_proba(X)
displayROCurve(y, y_prob, clf.classes_, 'BaggingClassifier')

# 5.超参优化（略）
# 6.使用模型（略）


# 相关类
#   1、BaggingClassifier(base_estimator=None, bootstrap=True, 
#               bootstrap_features=False, max_features=1.0, 
#               max_samples=1.0, n_estimators=10,
#               n_jobs=None, oob_score=False, random_state=None, 
#               verbose=0, warm_start=False)
#   2、BaggingRegressor()--参数列表和分类完全一样

# 重要参数
    # base_estimator : 基学习器（默认=None），默认为决策树。
    # n_estimators : int，基学习器的个数，（默认值为10）
# 抽样参数
    # max_samples : int或float，（默认值= 1.0），抽取样本的数量或比例
    #   如果为int，则抽取样本 max_samples。
    #   如果float，则抽取本 max_samples * X.shape[0]
    # max_features : int或float，（默认值= 1.0），抽取特征的数量或比例
    #   如果为int，则绘制特征 max_features。
    #   如果是浮动的，则绘制特征 max_features * X.shape[1]
    # bootstrap : 布尔值（默认= True），是否有放回抽取样本
    # bootstrap_features：布尔值（默认= False），是否有放回抽取特征
# 评估参数
    # oob_score : 布尔变量（默认为False），是否用袋外子集来评估泛化误差
    #   设为True，才能够访问模型的属性oob_score_
# 其它参数
    # warm_start : 布尔变量（默认= False）
    #   设置为True时，表示继续接着集成模型，添加更多学习器
    #   否则，重新建立新的模型。
    # n_jobs : int或None（默认为None），并行作业数。
    # random_state : 整型，RandomState实例或无，可选（默认值：无）
    #   如果为int，则random_state是随机数生成器使用的种子；否则为false
    # verbose：整型（默认0），控制复杂度。
    #       Controls the verbosity when fitting and predicting.

# 返回的属性
    # base_estimator：estimator,基学习器
    # n_features：int,特征数量
    # estimators_：list of estimators,已经训练好的基模型的集合
    # estimators_samples_：list of arrays每个基模型使用的数据子集
    # estimators_features_:list of arrays每个基模型使用的特征子集
    # classes_:ndarray of shape (n_classes,)因变量取值列表，即标签（分类才有）
    # n_classes_：int or list，标签的个数（分类才有）
    # oob_score_：float，袋外子集的评估得分，要求oob_score=True
    # oob_decision_function_：ndarray of shape (n_samples, n_classes)
    #       分类才有：返回袋外子集的类别预测概率，要求oob_score=True

    # oob_prediction_：ndarray of shape (n_samples,) 
    #       回归才有：使用袋外数据评估的预测值，要求oob_score=True

######################################################################
########  Part2、投票算法：VotingClassifier
######################################################################
from sklearn.ensemble import VotingClassifier

mdl1 = DecisionTreeClassifier(max_depth=4)
mdl2 = KNeighborsClassifier(n_neighbors=5)
mdl3 = SVC(kernel='rbf', probability=True)

clf = VotingClassifier(
            estimators=[('dtc', mdl1), ('knc', mdl2), ('svc', mdl3)],
            flatten_transform = False,  #默认True
            voting='soft')
clf.fit(X, y)

# 评估
y_pred =clf.predict(X)
displayClassifierMetrics(y, y_pred, clf.classes_,poslabel)

# #可以用名字来访问训练好的基学习器
# mdl4 = clf.named_estimators_['dtc']
# mdl5 = clf.estimators_[0]

# 同时返回三个基学习器的预测结果
pred = clf.transform(X)
# print(pred.shape)

# 函数transform的返回格式
# 若voting='hard'，二元组(2464, 3)
#   则返回预测结果，形如(n_samples, n_classifiers)
# 若voting='soft',则：
    # 若flatten_transform=True，
    #   则返回概率，形如(n_samples, n_classifiers* n_classes)
    # 二元组(2464, 6)，列分别是[第1个基类预测0的概率，第1个基类预测1的概率，第2个...,第3个基类预测1的概论]
    # 若flatten_transform=False，三元组(3, 2464, 2)
    #   则返回概率，形如(n_classifiers, n_samples, n_classes)
    # [0,30,0]表示：第0个基类，

# 投票类
    # VotingClassifier(estimators, *, voting='hard', 
    #           weights=None, n_jobs=None, 
    #           flatten_transform=True, verbose=False)
    # 重要参数：
    # estimators：list of (str, estimator) tuples指定学习器列表
    # voting='hard',{‘hard’, ‘soft’}投票策略
    # weights=None,array-like of shape (n_classifiers,)基学习器权重
    # n_jobs:int, default=None
    # flatten_transform：bool, default=True。
    #   指定transform函数的返回格式
    #   True时，返回矩阵(n_samples, n_classiffers*n_classes)
    #   False时，返回(n_classifiers, n_samples, n_classes)
    # verbosebool, default=False

    # 返回属性
    # estimators_:list of classifiers
    # named_estimators_:Bunch,Attribute to access any fitted sub-estimators by name.
    # classes_:array-like of shape (n_predictions,)

######################################################################
########  Part3、超参优化：VotingClassifier/GridSearchCV
######################################################################
# 重点是超参名称的构造形式

from sklearn.model_selection import GridSearchCV

mdl1 = DecisionTreeClassifier(max_depth=4)
mdl2 = KNeighborsClassifier(n_neighbors=5)
mdl3 = SVC(kernel='rbf', probability=True)

clf = VotingClassifier(
            estimators=[('dtc', mdl1), 
                        ('knc', mdl2), 
                        ('svc', mdl3)],
            voting='hard')
clf.fit(X, y)

params = {'dtc__max_depth':[4, 5, 8],
        'knc__n_neighbors':[5, 7, 9],
        'svc__kernel':['rbf','sigmoid'],
        'svc__C':[1.0, 2.0, 3.0]}
grid = GridSearchCV(estimator=clf, param_grid=params)
grid.fit(X, y)

# 其余类似（略）
print('最优超参：\n', grid.best_params_)


######################################################################
########  Part4、Bagging回归优化算法：BaggingRegressor
######################################################################

# 1、读入数据
filename = '回归分析.xlsx'
sheetname = '销售额'
df = pd.read_excel(filename, sheetname)
# print(df.columns.tolist())


######################################################################
########  Part4、Bagging回归优化算法：VotingRegressor
######################################################################

# VotingRegressor(estimators, *, weights=None, 
    #           n_jobs=None, verbose=False)
    #           默认采用加权平均值作为预测结果


