#-*- coding: utf-8 -*-

########  本文件实现判别分析模型，包括
# Part1、二分类LDA
# Part2、多分类LDA
# Part3、LDA用于降维
# Part4、二次判别QDA
# 
##################################

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from common import displayClassifierMetrics
from common import displayROCurve

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


######################################################################
########  Part1、二分类LDA
######################################################################

# 1.数据集
filename = '分类预测.xls'
sheet = '贷款违约'
df = pd.read_excel(filename, sheet)

df.drop('学历', axis=1, inplace=True)
# print(df.columns.tolist())

# 2.特征工程
cols = ['年龄', '工龄', '地址', '收入', '负债率', '信用卡负债', '其他负债']
target = '违约'

X = df[cols]
y = df[target]
posLabel = '是'

# 3.训练模型
mdl = LinearDiscriminantAnalysis(solver='svd')
mdl.fit(X, y)

# 当为二分类时，虽然mdl.classes_有二个值，但只有一个方程
# 映射直线
sr = pd.Series(data=[mdl.intercept_[0]] + mdl.coef_[0].tolist(),
        index=['常数'] + cols)
print(sr)

# 4、评估模型
y_pred = mdl.predict(X)
displayClassifierMetrics(y, y_pred, mdl.classes_,pos_label='是')

y_prob = mdl.predict_proba(X)
displayROCurve(y, y_prob, mdl.classes_)

# 6.应用模型
# 1)预测值
srPred = pd.Series(data=y_pred, index=df.index, name='预测值')

# 2）预测概率
dfProb = pd.DataFrame(data=y_prob, index=df.index, columns=mdl.classes_)

# 3）映射后的值
X_ = mdl.transform(X)
facts = ['f{}'.format(i+1) for i in range(X_.shape[1])]
dfFacts = pd.DataFrame(data=X_, index=df.index, columns=facts)

# 4)合并到原数据集
dfNew = pd.concat([y, dfFacts, srPred, dfProb], axis=1)
print(dfNew)

######################################################################
########  Part2、多分类LDA
######################################################################

# 1.读数据集
from sklearn import datasets

iris = datasets.load_iris()

X = pd.DataFrame( data = iris['data'], columns = iris['feature_names'] )
cols = X.columns.tolist()

y = pd.Series(data = iris['target'], name = 'kind')
target = y.name

labels = iris['target_names']
for i, lbl in enumerate(labels):
    y.replace(i, lbl, inplace=True)

df = pd.concat( [X, y], axis=1)

# 2.特征工程（略）

# 3.训练模型
mdl = LinearDiscriminantAnalysis(solver='svd')
mdl.fit(X, y)

# 1）判别方程
for i in range(len(mdl.classes_)):
    print('\n类别-{}-判别方程：'.format(mdl.classes_[i]))
    sr = pd.Series(data=[mdl.intercept_[i]] + mdl.coef_[i].tolist(),
        index=['常数'] + cols)
    print(sr)

# 2）降维后每个维度的方差解释比例
nfact = len(mdl.explained_variance_ratio_)
facts = ['f{}'.format(i+1) for i in range(nfact)]
sr = pd.Series(data=mdl.explained_variance_ratio_, index=facts)
print(sr)
sr.plot(kind='bar', title='因子的方差解释比例')

# 4、评估模型
y_pred = mdl.predict(X)
displayClassifierMetrics(y, y_pred, mdl.classes_)

y_prob = mdl.predict_proba(X)
displayROCurve(y, y_prob, mdl.classes_)

# 6.应用模型
# 1)预测值
srPred = pd.Series(data=y_pred, index=df.index, name='预测值')

# 2）预测概率
dfProb = pd.DataFrame(data=y_prob, index=df.index, columns=mdl.classes_)

# 3）映射后的值
X_ = mdl.transform(X)
facts = ['f{}'.format(i+1) for i in range(X_.shape[1])]
dfFacts = pd.DataFrame(data=X_, index=df.index, columns=facts)

# 4)合并到原数据集
dfNew = pd.concat([y, dfFacts, srPred, dfProb], axis=1)
print(dfNew)

######################################################################
########  Part3、线性判别分析LDA，用于降维
######################################################################

# 1.读取数据集
# 同上 iris

# 2.降维，因子数n_components
# 因子数必须<= min(n_classes - 1, n_features)
mdl = LinearDiscriminantAnalysis(n_components=2)
X_ = mdl.fit_transform(X, y)
print(X_.shape)

# 降维后每个因子的方差解释比例
fcts = ['f{}'.format(i+1) for i in range(len(mdl.explained_variance_ratio_))]
sr = pd.Series(data = mdl.explained_variance_ratio_, index = fcts)
print(sr)
sr.plot(kind='bar', title='因子解释的方差比例')

# 返回的判别方程
for i, lbl in enumerate(mdl.classes_):
    print('\n类别-{}的判别方程：'.format(lbl))
    sr = pd.Series(name=lbl,
                data = [mdl.intercept_[i]]+mdl.coef_[i].tolist(),
                index = ['常数']+cols)
    print(sr)

# 可视化，降维后的分类结果
plt.figure()
colors = ['navy', 'turquoise', 'darkorange']

for i, lbl in enumerate(mdl.classes_):
    plt.scatter(X_[y==lbl, 0], X_[y==lbl, 1], 
                color = colors[i], 
                alpha = 0.8, label=lbl)

plt.legend(loc='best', shadow=False, scatterpoints=1)
plt.title('LDA of IRIS dataset')
plt.show()


######################################################################
########  Part4、二次判别分析QDA
######################################################################

# 3.训练模型
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

mdl = QuadraticDiscriminantAnalysis()
mdl.fit(X, y)

# 4.评估模型
y_pred = mdl.predict(X)
displayClassifierMetrics(y, y_pred, mdl.classes_)

y_prob = mdl.predict_proba(X)
displayROCurve(y, y_prob, mdl.classes_)

# 其余略

# 相关类
    # class sklearn.discriminant_analysis.LinearDiscriminantAnalysis(solver=’svd’, shrinkage=None, priors=None, 
    #           n_components=None, store_covariance=False, tol=0.0001)
    # 参数：
    # solver：一个字符串，指定了求解最优化问题的算法，可以为如下的值。
        # 'svd'：奇异值分解。对于有大规模特征的数据，推荐用这种算法。
        # 'lsqr'：最小平方差，可以结合skrinkage参数。
        # 'eigen' ：特征分解算法，可以结合shrinkage参数。
    # skrinkage：字符串‘auto’或者浮点数活者None。该参数通常在训练样本数量小于特征数量的场合下使用。
    # 该参数只有在solver=lsqr或者eigen下才有意义
        # '字符串‘auto’：根据Ledoit-Wolf引理来自动决定shrinkage参数的大小。
        # 'None：不使用shrinkage参数。
        # 浮点数（位于0~1之间）：指定shrinkage参数。
    # priors：一个数组，数组中的元素依次指定了每个类别的先验概率。如果为None，则认为每个类的先验概率都是等可能的。
    # n_components：一个整数。指定了数组降维后的维度（该值必须小于n_classes-1）。
    # store_covariance：一个布尔值。如果为True，则需要额外计算每个类别的协方差矩阵。
    # warm_start：一个布尔值。如果为True，那么使用前一次训练结果继续训练，否则从头开始训练。
    # tol：一个浮点数。它指定了用于SVD算法中评判迭代收敛的阈值。
    
    # 返回值
    # coef_：权重向量。
    # intercept：b值。
    # covariance_：一个数组，依次给出了每个类别烦人协方差矩阵。
    # means_：一个数组，依次给出了每个类别的均值向量。
    # xbar_：给出了整体样本的均值向量。
    # n_iter_：实际迭代次数。
    
