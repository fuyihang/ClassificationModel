#-*- coding: utf-8 -*-

########  本文件实现决策树模型，包括
# Part1、二分类SVC
# Part2、多分类SVC
###################################
# 类        用途                关键参数
# SVC       样本少于10000时     C,kernel,degree,gamma
# NuSVC     与SVC类似           nu,kernel,degree,gamma
# LinearSVC 样本多于1000时      C,Penalty,loss
# SVR       回归问题            C,kernel,degree,gamma,epsilon
# NuSVR     与SVR类似           nu,C,kernel,degree,gamma
# OneClassSVM   异常检测        nu,kernel,degree,gamma

import pandas as pd
import numpy as np

from common import displayClassifierMetrics
from common import displayROCurve

######################################################################
########  Part1、二分类SVC
######################################################################

# 1.读数据集
filename = '分类预测.xls'
sheet='贷款违约'
df = pd.read_excel(filename,sheet)

# 2.特征工程
# 1）特征标识
intCols = ['年龄','工龄','地址','收入','负债率','信用卡负债','其他负债']
catCols = ['学历']

target = '违约'
y=df[target]
posLabel = '是'

# 2）数值变量->标准化
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_ = scaler.fit_transform(df[intCols])

dfInts = pd.DataFrame(X_, columns=intCols)

# 3)类别变量->哑变量
from sklearn.preprocessing import OneHotEncoder

enc = OneHotEncoder(sparse=False, drop='first')
X_ = enc.fit_transform(df[catCols])

cols = []
for cats in enc.categories_:
    cols.extend(cats[1:])

dfCats = pd.DataFrame(X_, columns=cols)

# 4）合并

X = pd.concat([dfInts, dfCats], axis=1)
cols = X.columns.tolist()

# 3.训练模型
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV

params = [{'C':np.linspace(0.01, 2, 10),
            'kernel':['linear','rbf','sigmoid'],
            'gamma':['scale','auto']},
        {'C':np.linspace(0.01, 2, 10),
            'kernel':['poly'],
            'degree':range(2,10),
            'gamma':['scale','auto']}]
# params = {'C':np.linspace(0.01, 2, 10),
#             'kernel':['linear','rbf','sigmoid'],
#             'gamma':['scale','auto']
#             }
mdl = SVC(probability=True)     #表示要计算概率
grid = GridSearchCV(mdl, param_grid=params)
grid.fit(X, y)

print('best param:', grid.best_params_)
print('best score:', grid.best_score_)
mdl = grid.best_estimator_
print(mdl.fit_status_)  #模型状态：0表示成功

# 查看模型信息
# 类别标签
labels = mdl.classes_
# 每个类别的支持向量的个数
vtsNum = mdl.n_support_

# 支持向量列表
vts = mdl.support_vectors_
# 支持向量的索引列表
idxs = mdl.support_

# 当kernel='linear'时，才有如下属性
if mdl.kernel == 'linear':
    print(mdl._intercept_)
    print(mdl.coef_)

# 4.评估模型
y_pred = mdl.predict(X)
displayClassifierMetrics(y, y_pred, mdl.classes_, posLabel)

y_prob = mdl.predict_proba(X)
displayROCurve(y, y_prob, mdl.classes_, '支持向量机')

# 6.应用模型（略）

######################################################################
########  Part2、多分类SVC
######################################################################
# SVC和NuSVC为多分类实现了'one-vs-one'的方法，从而训练n*(n-1)/2个模型。

# 1.读取数据
from sklearn import datasets

iris = datasets.load_iris()
cols = ['花萼长度','花萼宽度','花瓣长度','花瓣宽度']
labels = ['山鸢尾', '杂色鸢尾', '维吉尼亚鸢尾']
# cols = iris['feature_names']
# labels = iris['target_names']

# 2.特征工程
X = iris['data']
y = iris['target']

# 3.训练模型
from sklearn.svm import SVC
mdl = SVC(kernel = 'linear', decision_function_shape='ovr', probability=True)
mdl.fit(X, y)

# 4.评估模型
y_pred = mdl.predict(X)
displayClassifierMetrics(y, y_pred, mdl.classes_)

y_prob = mdl.predict_proba(X)
displayROCurve(y, y_prob, mdl.classes_, '支持向量机')

# 相关类
    # SVC(*, C=1.0, kernel='rbf', degree=3, gamma='scale', coef0=0.0, 
    # shrinking=True, probability=False, tol=0.001, cache_size=200, 
    # class_weight=None, verbose=False, max_iter=-1, 
    # decision_function_shape='ovr', break_ties=False, 
    # random_state=None)

    # C：正则项参数,C>0
        # C越大，相当于惩罚松弛变量，希望松弛变量接近0，即对误分类的惩罚增大，训练集时准确，但泛化能力弱。
        # C值小，对误分类的惩罚减小，允许容错，将他们当成噪声点，泛化能力较强。
    # kernel ：核函数，默认是rbf，可以是{‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’, ‘precomputed’}
        # linear – 线性核函数：u'v
        # poly – 多项式核函数：(gamma*u'*v + coef0)^degree
        # rbf – RBF高斯核函数：exp(-gamma|u-v|^2)
        # sigmoid - S曲线：tanh(gamma*u'*v + coef0)
        # precomputed - 预计算
        # callable 
    # degree ：多项式poly函数的阶数，默认是3，仅用于kernel='poly'
    # gamma ： {'scale','auto'} 
        # scale, 取值1/(n_features*X.var())
        # auto，取值1/n_features
    # coef0 ：核函数的常数项。对于‘poly’和 ‘sigmoid’有用。
    # probability ：是否启用概率估计.默认为False。 
        # 由于使用交叉验证，所以会导致训练变慢。
        # 为True时，才能够调用predict_proba函数，有可能与predict函数结果不一致
    # shrinking ：是否采用shrinking heuristic方法，默认为true
    # tol ：停止训练的误差值大小，默认为1e-3
    # cache_size ：核函数cache缓存大小(MB)，默认为200
    # class_weight ：类别的权重，字典形式传递。设置第几类的参数C为weight*C(C-SVC中的C)
    # verbose ：允许冗余输出
    # max_iter ：最大迭代次数。-1为无限制。
    # decision_function_shape ：{‘ovr’，‘ovo’}，默认ovr,多分类时用。
    # random_state ：伪随机数的种子，int值。用于评估概率时。

# 超参优化(C,kernal,degree, gamma,)
    # kernel = 'linear'时
    # kernel = 'poly'时
        # degree
    # 通用参数：
        # C,gamma,
    # 多分类：
        # decision_function_shape
