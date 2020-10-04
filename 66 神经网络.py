#-*- coding: utf-8 -*-

########  本文件实现决策树模型，包括
# Part1、MLPClassifier（连续自变量）
# Part2、MLPClassifier（类别自变量）
# Part3、实战：通信套餐精准推荐
###################################

import pandas as pd
import numpy as np

from common import displayClassifierMetrics
from common import displayROCurve

######################################################################
########  Part1、MLPClassifier（连续自变量）
######################################################################


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

# MLP对特征的缩放比较敏感，所以先标准化处理
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X = scaler.fit_transform(X)

# 3.训练模型
from sklearn.neural_network import MLPClassifier

layer_sizes=(10,5)  #(4,10,5,3)，输入输出自动补齐
mdl = MLPClassifier(hidden_layer_sizes=layer_sizes)
mdl.fit(X, y)

# 查看网络信息
# mdl.coefs_ 包含了构建模型的权值矩阵
# mdl.intercepts_  包含一系列偏置向量
# lt = [coef.shape for coef in mdl.coefs_]  #查看每层的节点的输入输出

print('网络层数：', mdl.n_layers_)
print('输出节点数：', mdl.n_outputs_)
for i, coefs in enumerate(mdl.coefs_):
    nodes = len(mdl.intercepts_[i])
    print('中间层{},节点数:{}'.format(i+1, nodes))
    for j in range(nodes):
        wt = [mdl.intercepts_[i][j]] + coefs[:,j].tolist()
        print('  节点{}:{}'.format(j+1, np.round(wt,2)))

# 4.评估模型
y_pred = mdl.predict(X)
displayClassifierMetrics(y, y_pred, mdl.classes_)

y_prob = mdl.predict_proba(X)
displayROCurve(y, y_prob, mdl.classes_, '神经网络')

# 5.超参优化

# 6.应用模型
# 1）保存模型
# 2）加载模型
# 3）预测

######################################################################
########  Part2、MLPClassifier（类别自变量）
######################################################################

# 1、读取数据
filename = '分类预测.xls'
sheet = '贷款违约'
df = pd.read_excel(filename, sheet)

# 2、特征工程
# 1）特征标识
intCols = ['年龄','工龄','地址','收入','负债率','信用卡负债','其他负债']
catCols = ['学历']

target = '违约'
y = df[target]
posLabel = '是'

# 2）类别变量-->哑变量
from sklearn.preprocessing import OneHotEncoder
enc = OneHotEncoder(drop='first', sparse=False)
X_ = enc.fit_transform(df[catCols])

cols = []
for cat in enc.categories_:
    print(cat[1:])
    cols.extend(cat[1:])

dfCats = pd.DataFrame(X_, columns=cols)

# 2）数值变量，要标准化
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_ = scaler.fit_transform(df[intCols])

dfInts = pd.DataFrame(X_, columns=intCols)

# 4）拼接数据集
X = pd.concat([dfCats, dfInts], axis=1)
cols = X.columns.tolist()

# 3、训练模型
from sklearn.neural_network import MLPClassifier
# 小数据集建议用lbfgs，收敛速度快；
# 大数据集建议用adam，收敛迅速，鲁棒性强。
# 如果学习率调整正确，使用sgd会更好。

mdl = MLPClassifier(activation='logistic',
        solver='lbfgs', 
        alpha=1e-5,
        learning_rate='constant',
        hidden_layer_sizes=(7,6,5), 
        random_state=1)
mdl.fit(X, y)

# 查看网络信息（同上，略）

# 4.评估模型
y_pred = mdl.predict(X)
displayClassifierMetrics(y, y_pred, mdl.classes_, posLabel)

y_prob = mdl.predict_proba(X)
displayROCurve(y, y_prob, mdl.classes_, '神经网络')

# 5.超参优化
from sklearn.model_selection import GridSearchCV

# params = [{'solver': 'sgd', 'learning_rate': 'constant', 'momentum': 0,
    #        'learning_rate_init': 0.2},
    #       {'solver': 'sgd', 'learning_rate': 'constant', 'momentum': .9,
    #        'nesterovs_momentum': False, 'learning_rate_init': 0.2},
    #       {'solver': 'sgd', 'learning_rate': 'constant', 'momentum': .9,
    #        'nesterovs_momentum': True, 'learning_rate_init': 0.2},
    #       {'solver': 'sgd', 'learning_rate': 'invscaling', 'momentum': 0,
    #        'learning_rate_init': 0.2},
    #       {'solver': 'sgd', 'learning_rate': 'invscaling', 'momentum': .9,
    #        'nesterovs_momentum': True, 'learning_rate_init': 0.2},
    #       {'solver': 'sgd', 'learning_rate': 'invscaling', 'momentum': .9,
    #        'nesterovs_momentum': False, 'learning_rate_init': 0.2},
    #       {'solver': 'adam', 'learning_rate_init': 0.01}]
params = {
    'hidden_layer_sizes': [(12, 5), (128,), (128, 7)],
    'activation':['identity', 'logistic','tanh','relu'],
    'solver':['lbfgs', 'sgd', 'adam'],
    'alpha':[0.1, 0.2, 0.8, 1.0] }
mdl = MLPClassifier(learning_rate='adaptive', learning_rate_init=1., early_stopping=True, shuffle=True)
grid = GridSearchCV(mdl, param_grid=params)
grid.fit(X, y)

print('最优超参：',grid.best_params_)

mdl = grid.best_estimator_
y_pred = mdl.predict(X)
displayClassifierMetrics(y, y_pred, mdl.classes_,posLabel)

y_prob = mdl.predict_proba(X)
displayROCurve(y, y_prob, mdl.classes_, '神经网络')

# 6.应用模型
# 1）保存模型
# 2）加载模型
# 3）预测


# 相关类别
#  MLPClassifier(activation='relu', alpha=1e-05, batch_size='auto',
    #  beta_1=0.9, beta_2=0.999, early_stopping=False,
    #  epsilon=1e-08, hidden_layer_sizes=(5, 2), learning_rate='constant',
    #  learning_rate_init=0.001, max_iter=200, momentum=0.9,
    #  nesterovs_momentum=True, power_t=0.5, random_state=1, shuffle=True,
    #  solver='lbfgs', tol=0.0001, validation_fraction=0.1, verbose=False,
    #  warm_start=False)
    # hidden_layer_sizes：元组，表示各个隐藏层的神经元个数
    # activation：{identity, logistic,tanh,relu}
    #   identity: f(x)=x
    #   logistic: f(x)=1/(1+exp(-x))
    #   tanh:     f(x)=tanh(x)
    #   relu:     f(x)=max(0,x)
    # solver:{lbfgs, sgd, adam} 优化权重
    #   lbfgs: quasi-Newton优化器，适合于小数据集，收敛快效果好
    #   sgd: 随机梯度下降
    #   adam:机遇随机梯度的优化器，适合于大数据集
    # alpha：作为正则化（L2正则化）系数，通过惩罚大数量级的权重值以避免过拟合
    # batch_size：随机优化的minibatches的大小。如果solver='lbfgs',应设置为'auto'.
    # learning_rate:{constant, invscaling,adaptive}，权重更新，当solver='sgd'时使用
    #   constant :恒定学习率
    #   invscaling:随着时间t使用power_t的逆标度指数不断降低学习率effective_learning_rate = learning_rate_init / pow(t, power_t) 
    #   adaptive:只要训练损耗在下降，就保持学习率为’learning_rate_init’不变，当连续两次不能降低训练损耗或验证分数停止升高至少tol时，将当前学习率除以5. 
    # learning_rate_init: 初始学习率。只有当solver为sgd或adam时才用。
    # shuffle: 判断是否在每次迭代时对样本进行清洗，当solver='sgd'或'adam'时使用
    # tol: 优化的容忍度
    # power_t： 逆扩展学习率的指数，更新有效学习率（当solver为sgd时才用）
    # shuffle:是否在每次迭代时对样本进行清洗（当solver为sgd或adam时才用）
    # warm_start:是否使用之前的解决方法作为初始拟合
    # momentum:梯度下降的冲量，0~1，solver为sgd时用
    # nesterovs_momentum:是否使用Nesterov冲量
    # early_stopping：当验证效果不再改善的时候是否终止训练，当为True时，自动选出10%的训练数据用于验证并在两步连续爹迭代改善低于tol时终止训练。 
    # 1validation_fraction :  0~1，default 0.1,用作早期停止验证的预留训练数据集的比例，只当early_stopping=True有用 
    # beta_1 : default 0.9，Only used when solver=’adam’，估计一阶矩向量的指数衰减速率，[0,1)之间 
    # beta_2 : default 0.999,Only used when solver=’adam’估计二阶矩向量的指数衰减速率[0,1)之间 
    # epsilon : default 1e-8,Only used when solver=’adam’数值稳定值。solver为adam时使用 

    # 可用属性
    # - classes_:每个输出的类标签 
    # - loss_:损失函数计算出来的当前损失值 
    # - coefs_:列表中的第i个元素表示i层的权重矩阵 
    # - intercepts_:列表中第i个元素代表i+1层的偏差向量 
    # - n_iter_ ：迭代次数 
    # - n_layers_:层数 
    # - n_outputs_:输出的个数 
    # - out_activation_:输出激活函数的名称。
