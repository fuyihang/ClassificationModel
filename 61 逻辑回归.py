#-*- coding: utf-8 -*-

########  本文件实现逻辑回归预测模型，包括
# Part1、二项逻辑回归
# Part2、带分类的逻辑回归
# Part3、超参优化
# Part4、多项逻辑回归
###################################

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from common import displayClassifierMetrics
from common import displayROCurve

######################################################################
########  Part1、二分类逻辑回归
######################################################################
# 目标变量只有二种取值

############################
######## 订阅者画像分析 ########
############################

# 1、读取数据
filename = '分类预测.xls'
sheet = '杂志订阅信息'
df = pd.read_excel(filename, sheet)
# print(df.columns.tolist())

# 2、数据预处理
# 1) 数据标题
cols = ['年龄', '收入']
target = '是否订阅'

X = df[cols]
y = df[target]
poslabel = '是'

# 2）可视化观察
# 画图观察
groupYes = df[df[target] == '是']
groupNo = df[ df[target]=='否']

plt.scatter(groupYes['年龄'], groupYes['收入'],c='g')
plt.scatter(groupNo['年龄'], groupNo['收入'],c='b')

plt.title('杂志订阅')
plt.xlabel('年龄')
plt.ylabel('收入')
plt.show()
# 蓝色点主要聚集在左下角


# 3、训练模型
from sklearn.linear_model import LogisticRegression

mdl = LogisticRegression(penalty='none')
mdl.fit(X, y)

# 注意：逻辑回归中的这两个参数都是元组，不像回归模型
sr = pd.Series(
        data=[mdl.intercept_[0]] + mdl.coef_[0].tolist(),
        index=['常数'] + cols )
print(sr)

####### 5、评估模型指标
y_pred = mdl.predict(X)
displayClassifierMetrics(y, y_pred, mdl.classes_, poslabel)

y_prob = mdl.predict_proba(X)
displayROCurve(y, y_prob, mdl.classes_, '二分类逻辑回归')

# 6.应用模型

# 1）可选操作：将原始数据，预测值，预测概率合并在一个DF集
srPred = pd.Series(y_pred, index=df.index, name='预测值')            #预测结果

# y_prob有顺序与classes_指定的类别顺序是一致的
ProbCols = [F"{val}-概率" for val in mdl.classes_]
dfProb = pd.DataFrame(y_prob, index=df.index, columns=ProbCols)

dfNew = pd.concat([df, srPred, dfProb], axis=1)
print(dfNew.head())

# 2）保存模型（略）
# 3）加载模型（略）
# 4）可视化
plt.figure()
plt.title('杂志订阅')
plt.xlabel('年龄')
plt.ylabel('收入')

X1 = df[y=='是']
X2 = df[y=='否']

plt.scatter(X1['年龄'], X1['收入'], color='r', alpha = 0.8, label='是')
plt.scatter(X2['年龄'], X2['收入'], color='b', alpha = 0.8, label='否')

# 增加回归线:f(x) = 常数 + a*年龄 + b*收入 = 0
ages = np.linspace(np.min(df['年龄']), np.max(df['年龄']), 20)
# incomes = [(-mdl.intercept_[0] - mdl.coef_[0,0]*age)/mdl.coef_[0,1] for age in ages]
incomes = [(- sr['常数'] - sr['年龄']*age)/sr['收入'] for age in ages]

plt.plot(ages, incomes, color='black')
plt.show()

############################
######## 练习：上市公司类型分析 ########
############################

# 1、读取数据
filename = '分类预测.xls'
sheetname = '上市公司类型'
df = pd.read_excel(filename,sheetname)
# print(df.columns.tolist())

# 2、数据处理
cols = ['流动比率', '资产周转率', '资产净利率', '资产增长率']
target = '类型'

X = df[cols]
y = df[target]
poslabel = 'ST'

# 3、训练模型
# 我这里尝试用L1正则项，实现自动特征选择。
mdl = LogisticRegression(penalty='l1', 
            solver='liblinear',C=1)
mdl.fit(X, y)

sr = pd.Series(
        data=[mdl.intercept_[0]] + mdl.coef_[0].tolist(),
        index=['常数'] + cols )
print(sr)

# 4、评估模型
y_pred = mdl.predict(X)
displayClassifierMetrics(y, y_pred, mdl.classes_, poslabel)

y_prob = mdl.predict_proba(X)
displayROCurve(y, y_prob, mdl.classes_)

######################################################################
########  Part2、带分类的逻辑回归
######################################################################

# 1、读取数据
filename = '分类预测.xls'
sheet = '是否购买'
df = pd.read_excel(filename,sheet)
# print(df.columns.tolist())


# 2、特征工程
# 1）特征标识
intCols = ['年龄']
catCols =['性别', '收入']

target = '是否购买'
y = df[target]
poslabel = '是'

# 2）哑变量化
from sklearn.preprocessing import OneHotEncoder

enc = OneHotEncoder(drop='first', sparse=False)
X_ = enc.fit_transform(df[catCols])

cols = []
for i in range(len(enc.categories_)):
    cols.extend(enc.categories_[i][1:])

dfCats = pd.DataFrame(data=X_, index=df.index,columns=cols)

# 3）合并数据集
X = pd.concat([dfCats, df[intCols]], axis=1)
cols = X.columns.tolist()

# 3、训练模型
# 我这里尝试用L1正则项，实现自动特征选择。
mdl = LogisticRegression()
mdl.fit(X, y)

sr = pd.Series(
        data=[mdl.intercept_[0]] + mdl.coef_[0].tolist(),
        index=['常数'] + cols )
print(sr)

# 4、评估模型
y_pred = mdl.predict(X)
displayClassifierMetrics(y, y_pred, mdl.classes_, poslabel)

y_prob = mdl.predict_proba(X)
displayROCurve(y, y_prob, mdl.classes_)

######################################################################
########  Part3、超参数优化 hyperparameter
######################################################################

# 1、读入数据
filename = '分类预测.xls'
sheet = '贷款违约'
df = pd.read_excel(filename,sheet)

# 2、特征工程
# 1）特征标识
intCols = ['年龄', '工龄', '地址', '收入', '负债率', '信用卡负债', '其他负债']
catCols = ['学历']

target = '违约'
y = df[target]
poslabel = '是'

# 2）类别变量哑变量化
from sklearn.preprocessing import OneHotEncoder
enc = OneHotEncoder(drop='first', sparse=False)
X_ = enc.fit_transform(df[catCols])

cols = []
for i in range(len(enc.categories_)):
    cols.extend(enc.categories_[i][1:])

dfCats = pd.DataFrame(data=X_, index=df.index, columns=cols)

# 3）合并数据
X = pd.concat([dfCats, df[intCols]], axis=1)
cols = X.columns.tolist()

############################
######## 1）使用交叉验证LogisticRegressionCV
############################
# 参数Cs的取值说明：
#   --float列表，对应于前面的C取值列表。比如：Cs=[0.1,1,10,100]
#   --整数，则指定在1e4和1e4之间的取的个数。相当于C=np.linspace(1e-4, 1e4, CS)
#           生成的列表可以通过mdl.Cs_来观看
# 速度快，可惜，这种方式只能调优C和l1_ratio两个参数

from sklearn.linear_model import LogisticRegressionCV

Cs = [0.1,1,10,100, 10000]
ratios = np.arange(0.1, 1.0, 0.1)   #0.1~0.9的值
mdl = LogisticRegressionCV(Cs=Cs, l1_ratios=ratios, 
            scoring='accuracy', cv = 5, 
            penalty='elasticnet', solver='saga')
mdl.fit(X, y)

print('最优C值：',mdl.C_[0])
print('最优l1_ratio值：', mdl.l1_ratio_[0])

sr = pd.Series(
        data=[mdl.intercept_[0]] + mdl.coef_[0].tolist(),
        index=['常数'] + cols )
print(sr)

####### 评估模型
y_pred = mdl.predict(X)
displayClassifierMetrics(y, y_pred, mdl.classes_, poslabel)

y_prob = mdl.predict_proba(X)
displayROCurve(y, y_prob, mdl.classes_)

############################
######## 2）使用网络搜索GridSearchCV
############################
# 速度慢。
# 考虑到penalty、solver、l1_ratios之间存在某种约束

from sklearn.model_selection import GridSearchCV

# 2、构造参数字典
Cs = [0.1, 2.0, 5.0]
ratios = np.arange(0.1, 1.0, 0.1)   #0.1~0.9的值

params = [{'C':Cs,
            'penalty':['l1'],
            'solver':['liblinear','saga']},
        #l1时，只支持两种算法：liblinear','saga
        {'C':Cs,
            'penalty':['l2'],
            'solver':['newton-cg','lbfgs','sag','saga']},
        # l2时，支持四种算法
        {'C':Cs, 
            'l1_ratio':ratios,
            'penalty':['elasticnet'],
            'solver':['saga']}
        # elasticnet时，只支持saga
        ]

# 3、训练模型
mdl = LogisticRegression(max_iter=1000)
grid = GridSearchCV(mdl, param_grid=params,scoring='accuracy',cv=5)
grid.fit(X, y)

print('最优超参：\n',grid.best_params_)
mdl = grid.best_estimator_

sr = pd.Series(
        data=[mdl.intercept_[0]] + mdl.coef_[0].tolist(),
        index=['常数'] + cols )
print(sr)

####### 评估模型
y_pred = mdl.predict(X)
displayClassifierMetrics(y, y_pred, mdl.classes_, poslabel)

y_prob = mdl.predict_proba(X)
displayROCurve(y, y_prob, mdl.classes_)

######################################################################
########  Part4、多分类逻辑回归
######################################################################
# 实现原理，将多分类（K分类）转化成二分类
# 两种方式：
# ovr:one vs rest
#   1）将其中一类当成正类，其余所有类当成负类
#   2）建立K个逻辑回归模型，计算每个样本的类别概率
#   3）预测时，选取概率最大的类别当成预测结果

# ovo:one vs one,
#   1）依次选取每两个类，建立二分类逻辑回归
#   2）建立K*(K-1)个逻辑回归模型，计算每个样本的类别
#   3）预测时，所有模型投票，得票多的类别当成预测结果


# 1、准备数据

from sklearn import datasets
# Iris鸢尾花卉数据集，由Fisher, 1936收集整理。
# 数据集包含150个数据样本，分为3类，每类50个数据，每个数据包含4个属性。
# Sepal.Length, Sepal.Width, Petal.Length, Petal.Width
# 三个类别：Setosa（山鸢尾-0），Versicolour（杂色鸢尾-1），Virginica（维吉尼亚鸢尾-2）

iris = datasets.load_iris()
cols = iris['feature_names']
labels = iris['target_names']
X = iris['data']
y = iris['target']

# 换成中文标题
cols = ['花萼长度','花萼宽度','花瓣长度','花瓣宽度']
target = '种类'

X = pd.DataFrame(data=X, columns=cols)
y = pd.Series(data=y, name = target)
y.replace(0,'山鸢尾', inplace=True)
y.replace(1,'杂色鸢尾', inplace=True)
y.replace(2,'维吉尼亚鸢尾', inplace=True)
df = pd.concat([X, y], axis=1)

# 2.特征工程（略）

# 3.训练模型
from sklearn.linear_model import LogisticRegression

# mdl = LogisticRegression(penalty='l1', solver='liblinear', 
#         multi_class='ovr')
mdl = LogisticRegression(penalty='l2', solver='lbfgs', 
      multi_class='multinomial')
mdl.fit(X, y)

# 注意：
# k分类中，常数项/回归系数列表都应该有k个
for i in range(len(mdl.classes_)):
    print('类别-{}-的回归系数：'.format(mdl.classes_[i]))
    print('常数:', mdl.intercept_[i])
    print('系数：\n', mdl.coef_[i])

# 4、评估模型
y_pred = mdl.predict(X)
displayClassifierMetrics(y, y_pred, mdl.classes_)

y_prob = mdl.predict_proba(X)
displayROCurve(y, y_prob, mdl.classes_)

# 5、模型优化（略）
# 6、应用模型（略）


######################################################################
########  Part5、实战案例
######################################################################
# 预测套餐类型，实现精准套餐推荐


# 1、读取数据集
filename = 'Telephone.csv'
df = pd.read_csv(filename, encoding='gbk')
# print(df.columns.tolist())

# 2、数据处理
# 1）特征标识
intCols = ['年龄','收入','家庭人数','开通月数']
catCols = ['居住地', '婚姻状况', '教育水平', '性别', '电子支付']

target = '套餐类型'
y = df[target]

# 2）哑变量化
from sklearn.preprocessing import OneHotEncoder

enc = OneHotEncoder(drop='first', sparse=False)
X_ = enc.fit_transform(df[catCols])

cols = []
for i in range(len(enc.categories_)):
    cols.extend(enc.categories_[i][1:])

dfCats = pd.DataFrame(data=X_, index=df.index,columns=cols)

# 3）合并数据集
X = pd.concat([dfCats, df[intCols]], axis=1)
cols = X.columns.tolist()

# 3、训练模型
from sklearn.linear_model import LogisticRegression

mdl = LogisticRegression(penalty='l1', solver='liblinear', 
        multi_class='ovr')
# mdl = LogisticRegression(penalty='l2', solver='lbfgs', 
#       multi_class='multinomial')
mdl.fit(X, y)

# 注意：
# k分类中，常数项/回归系数列表都应该有k个
for i in range(len(mdl.classes_)):
    print('\n类别-{}-的回归系数：'.format(mdl.classes_[i]))
    print('常数:', mdl.intercept_[i])
    print('系数：\n', mdl.coef_[i])

# 4、评估模型
y_pred = mdl.predict(X)
displayClassifierMetrics(y, y_pred, mdl.classes_)

y_prob = mdl.predict_proba(X)
displayROCurve(y, y_prob, mdl.classes_)

# 5、模型优化（略）
# 6、应用模型（略）


# filename = 'out.csv'
# dfNew.to_csv(filename, encoding='gbk', index=False)

# sklearn.linear_model.LogisticRegression(penalty=’l2’, dual=False, tol=0.0001, 
    #       C=1.0, fit_intercept=True, intercept_scaling=1, class_weight=None, 
    #       random_state=None, solver=’warn’, max_iter=100, multi_class=’warn’, 
    #       verbose=0, warm_start=False, n_jobs=None, l1_ratio=None)
    # 重要参数
    #   solver:优化算法
    #   penalty:惩罚项
    #   惩罚项超参：C、l1_ration
    # 重要方法
    #   decision_function(self, X)	返回预测置信度
    #   fit(self, X, y[, sample_weight])	训练模型
    #   predict(self, X)	        返回预测结果标签
    #   predict_log_proba(self, X)	返回概率对数
    #   predict_proba(self, X)	    返回预测概率
    #   score(self, X, y[, sample_weight])	返回平均正确率（mean accuracy）
    #   set_params(self, \*\*params)	Set the parameters of this estimator.
    #   get_params(self[, deep])	Get parameters for this estimator.
    #   sparsify(self)	转换系数矩阵到稀疏格式
    #   densify(self)	转换系数矩阵到密集格式

# sklearn.linear_model.LogisticRegressionCV(Cs=10, fit_intercept=True, cv=’warn’, 
    #       dual=False, penalty=’l2’, scoring=None, solver=’lbfgs’, tol=0.0001, 
    #       max_iter=100, class_weight=None, n_jobs=None, verbose=0, refit=True, 
    #       intercept_scaling=1.0, multi_class=’warn’, random_state=None, l1_ratios=None)
    # 重要参数：
    # Cs：指定在[1e-4, 1e4]区间产生多少个C值，越小则正则化直强
    #       比如Cs=10，则产生这10个等距点
    #       再产生相应的10个C，[10^-4, 10^-3.11, ..., 10^4]，再从10个C里面挑出最佳的C。
    # l1_ratios: 弹性网络的超参[0,1]列表
    # cv：采用分层K折交叉，指定整数可指定折数，默认5折
    # tol: 默认1e-4，迭代终止的误差范围
    # refit:默认True，表示最后返回最优的参数和模型。
    # multi_class:多分类问题的处理方式
    #       ovr:
    #       multinomial:
    #       auto:自动选择。若数据是二分类或solver='liblinear'，则使用ovr；否则为multinomial
    # 