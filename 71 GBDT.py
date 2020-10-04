#-*- coding: utf-8 -*-

########  本文件实现GBDT模型，包括
# Part1、回归:GradientBoostingRegressor
# Part2、分类：GradientBoostingClassifier
# Part3、超参优化：/GridSearchCV
# Part4、基于直方图的GBDT：HistGradientBoostingClassifier
###################################

import pandas as pd
import numpy as np

from common import displayClassifierMetrics
from common import displayRegressionMetrics
from common import displayROCurve


from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import GradientBoostingRegressor

from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import GridSearchCV

######################################################################
########  Part1、梯度提升回归：GradientBoostingRegressor
######################################################################

# 1、读取数据
filename = 'Telephone.csv'
df = pd.read_csv(filename, encoding='gbk')
# print(df.columns.tolist())


# 2、数据处理
# 1)特征标识
intCols = ['年龄','收入','家庭人数','开通月数']
catCols = ['居住地','婚姻状况','教育水平','性别']
target = '消费金额'
y = df[target]

# 2）数字化
# 使用决策树，类别变量一定要数字化
from sklearn.preprocessing import OrdinalEncoder

enc = OrdinalEncoder(dtype='int')
X_ = enc.fit_transform(df[catCols])

dfCats = pd.DataFrame(X_, columns=catCols)

for i, col in enumerate(catCols):
    print('\n{}:{}'.format(col ,enc.categories_[i]))

# 3）合并
X = pd.concat([dfCats, df[intCols]], axis=1)
cols = X.columns.tolist()

# 3、训练模型
from sklearn.ensemble import GradientBoostingRegressor
mdl = GradientBoostingRegressor(
        n_estimators=100, learning_rate=1.0,
            max_depth=1, random_state=10
)
mdl.fit(X, y)

# 4、评估模型
y_pred = mdl.predict(X)
displayRegressionMetrics(y, y_pred, X.shape)

# 显示特征重要性
sr = pd.Series(mdl.feature_importances_, index = cols, name='特征重要性')
sr.sort_values(ascending=False, inplace=True)
sr.plot(kind='bar', title=sr.name)

print('特征总数：', mdl.n_features_)
print('使用特征数：', mdl.max_features_)

for i, est in enumerate(mdl.estimators_):
    # 可以单独保存/使用学习器est
    print('第{}个得分:{}'.format(i, mdl.train_score_[i]))

# 5.超参优化（略）
# 6.应用模型（略）


######################################################################
########  Part2、梯度提升分类：GradientBoostingClassifier
######################################################################
# 1、读取数据
filename = 'Telephone.csv'
df = pd.read_csv(filename, encoding='gbk')
# print(df.columns.tolist())

# 2、特征工具
# 1）特征标识/筛选
intCols = ['年龄','收入', '家庭人数', '开通月数']
catCols = ['居住地', '婚姻状况', '教育水平', '性别']

target = '流失'
y = df[target]
poslable = 'Yes'

# 2）类别变量数字化
from sklearn.preprocessing import OrdinalEncoder

enc = OrdinalEncoder(dtype='int')
X_ = enc.fit_transform(df[catCols])

dfCats = pd.DataFrame(X_, columns=catCols)

# 3）合并
X = pd.concat([dfCats, df[intCols]], axis=1)
cols = X.columns.tolist()

# 3、训练模型
from sklearn.ensemble import GradientBoostingClassifier

clf = GradientBoostingClassifier(
            loss='deviance',
            n_estimators=100, 
            learning_rate=1.0,
            subsample = 0.8,
            max_depth=1, 
            random_state=0)
clf.fit(X, y)

# 4、评估模型
print('score=', clf.score(X, y))

y_pred = clf.predict(X)
displayClassifierMetrics(y, y_pred, clf.classes_, poslable)

y_prob = clf.predict_proba(X)
displayROCurve(y, y_prob, clf.classes_, 'GBDT')

# 显示特征重要性
sr = pd.Series(mdl.feature_importances_, index = cols, name='特征重要性')
sr.sort_values(ascending=False, inplace=True)
sr.plot(kind='bar', title=sr.name)

# 5.超参优化（略）
# 6.应用模型（略）


# 保存模型（略）
# sklearn.ensemble.GradientBoostingRegressor
    # (loss='ls', learning_rate=0.1, n_estimators=100, 
    # subsample=1.0, criterion='friedman_mse', 
    # min_samples_split=2, min_samples_leaf=1, 
    # min_weight_fraction_leaf=0.0, max_depth=3, 
    # min_impurity_decrease=0.0, min_impurity_split=None, 
    # init=None, random_state=None, max_features=None, 
    # alpha=0.9, verbose=0, max_leaf_nodes=None, 
    # warm_start=False, presort='deprecated', 
    # validation_fraction=0.1, n_iter_no_change=None, 
    # tol=0.0001, ccp_alpha=0.0)

    # 框架参数
        # n_estimators=100,
        # learning_rate=0.1,学习率
        # loss='ls',{‘ls’, ‘lad’, ‘huber’, ‘quantile’}
            # 指定损失函数，决定了梯度的计算
            # ls:MSE，若噪音数据不多，使用ls
            # lad:MAE
            # huber:MSE+MAE，若噪音数据多，推荐huber
            # quantile:分位数，若需要对训练集进行分段预测的时候，推荐quantile
        # alpha=0.9, 当loss='huber'或'quantile'时使用。
            # 若噪音数据多，可适当降低这个值
    # 基学习器CART参数
        # criterion='friedman_mse','mse','mae'.
            # 指定CART树分割时用的判断准则
            # friedman_mse：使用MSE的优化
            # mse:使用MSE
            # mae:使用MAE
        # subsample=1.0, 抽样百分比。
            # 小于1时将采用随机梯度提升，会导致方差就小，偏差增大。
        # min_samples_split=2, min_samples_leaf=1,
        # min_weight_fraction_leaf=0.0,
        # max_depth=3, max_features=None, max_leaf_nodes=None,
        # min_impurity_decrease=0.0, min_impurity_split=None,
        # ccp_alpha=0.0 预剪枝参数
    # 其它参数
        # init=None, 初始预测值，即H0，默认为平均值，或者分位数。
        # random_state=None, 
        # verbose=0, 是否输出详细的过程信息。
        #  warm_start=False, 
        # validation_fraction=0.1, 早期停止时，作为验证集的百分比。当n_iter_no_change为整数时使用
        # n_iter_no_change=None, 当为整数时，表示迭代指定次数后将启用早期停止。
        # tol=0.0001, 
    # 重要属性
        # feature_importances_:ndarray of shape (n_features,)
        # oob_improvement_:ndarray of shape (n_estimators,)
            # 袋外数据的误差提升。当subsample < 1.0有效
        # train_score_:ndarray of shape (n_estimators,)训练得分（偏差）
        # loss_:LossFunction损失函数对象
        # init_:estimator.用于初始化的学习器，对应输入参数init
        # estimators_:ndarray of DecisionTreeRegressor of shape (n_estimators, 1)
        # n_features_:int
        # max_features_:int,对应输入参数max_features.

# sklearn.ensemble.GradientBoostingClassifier
    # (ccp_alpha=0.0, criterion='friedman_mse', init=None,
    #    learning_rate=0.1, loss='deviance', max_depth=3,
    #    max_features=None, max_leaf_nodes=None,
    #    min_impurity_decrease=0.0, min_impurity_split=None,
    #    min_samples_leaf=1, min_samples_split=2,
    #    min_weight_fraction_leaf=0.0, n_estimators=100,
    #    n_iter_no_change=None, presort='deprecated',
    #    random_state=None, subsample=1.0, tol=0.0001,
    #    validation_fraction=0.1, verbose=0,
    #    warm_start=False)

    # 基本与回归相同，差异如下：
    # 框架参数
        # loss: {‘deviance’, ‘exponential’}
            # deviance:和逻辑回归一样，对数似然
            # exponential:指数损失，和AdaBoost算法类似
    # 重要属性，新增：
        # n_classes_: int
        # classes_: ndarray of shape(n_classes,)

######################################################################
########  Part3、GBDT超参优化
######################################################################
# #########框架参数
# n_estimators, learning_rate, 
# subsample, init,loss, alpha
# 
# ###########CART回归树
# 同前(略)

# 注意事项：
# 1）首先调优框架超参
# 2）n_estimators和learning_rate一般一起调优
#       learning_rate越大越有可能过拟合，但越小会导致需要的n_estimators数量要多。
#       因此，一般选取一个相对稍高点的学习率，比如[0.05, 0.2]间，
#       再调优个数，n_estimators一般在[40-70]之间
# 3）在框架调参时，可以先确定其它参数的经验值
# 4）再调优基学习器--CART回归树的超参

from sklearn.model_selection import GridSearchCV

bestParams = {}
lt_params = [
    {'n_estimators':range(20,81,10),
        'learning_rate':np.linspace(0.01, 1.0,50)},
    {'max_depth':range(3,14,2),
        'max_features':range(7, 20, 2), 
        'min_samples_split':range(100, 801, 200)},
    {'min_samples_split':range(800, 1900, 200), 
        'min_samples_leaf':range(60, 101, 10)},
    {'subsample':[0.6, 0.7, 0.75, 0.8, 0.85, 0.9]}
]

mdl = GradientBoostingClassifier(
    learning_rate=0.1, 
    max_depth=8, 
    max_features='sqrt',
    min_samples_split=300,
    min_samples_leaf=20, 
    subsample=0.8,      #一般都建议抽样训练
    random_state=10)

for params in lt_params:
    grid = GridSearchCV(estimator = mdl, 
                       param_grid = params)
    grid.fit(X,y)

    bestParams.update(grid.best_params_)
    # mdl.set_params(**bestParams)

print('best_score=', grid.best_score_)
clf = grid.best_estimator_
y_pred = clf.predict(X)
displayClassifierMetrics(y, y_pred, clf.classes_)

y_prob = clf.predict_proba(X)
displayROCurve(y, y_prob, clf.classes_, 'GBDT')

# 显示特征重要性(略)


######################################################################
########  Part4、基于直方图的GBDT：HistGradientBoostingClassifier
######################################################################
# 
# # scikit-learn 0.23.1版本才有下面两个类
# from sklearn.ensemble import HistGradientBoostingClassifier
# from sklearn.ensemble import HistGradientBoostingRegressor

# from sklearn.ensemble import _hist_gradient_boosting

# sklearn.ensemble.HistGradientBoostingClassifier(
    # loss={‘auto’, ‘binary_crossentropy’, ‘categorical_crossentropy’}
        # 损失函数
        # ‘binary_crossentropy’ 对数似然，二分类
        # ‘categorical_crossentropy’ 多分类
    # l2_regularization=0.0,L2正则项系数
    # max_bins=255, 预处理时的最大分箱数.(0,255]
    # monotonic_cst=None, array-like of int of shape (n_features)
    #   指示特征的单调的约束.
    #   -1, 1 and 0分别表示正约束、负约束和无约束
    # *, learning_rate=0.1, 

    # 基学习器参数
    # max_leaf_nodes=31, max_depth=None, 
    # min_samples_leaf=20,  
    
    # 早期停止参数
        # early_stopping=‘auto’ or bool是否启用早期停止
            # auto时，当样本大于10000时自动启用
        # scoring='loss', str or callable or None
            # 用于早期停止的得分参数。
        # validation_fraction=0.1, int or float or None
            # 早期停止时使用的验证的样本数或百分比。
            # None表示在训练集上作早期停止
        # n_iter_no_change=10, 早期停止的次数。
        # tol=1e-07, 
    # 其它参数
        # max_iter=100,
        # warm_start=False, 
        # verbose=0, random_state=None)
    # 重要属性
        # classes_:array, shape = (n_classes,)
        # n_iter_:int,早期停止的迭代次数
        # n_trees_per_iteration_:int 每次迭代构建的树的个数，与类别个数有关
        # train_score_:ndarray, shape (n_iter_+1,)
        # validation_score_:ndarray, shape (n_iter_+1,)

# sklearn.ensemble.HistGradientBoostingRegressor(
    # 参数和分类基本相同，差异：
    # loss={‘least_squares’, ‘least_absolute_deviation’, ‘poisson’}
        # Note that the “least squares” and “poisson” losses actually 
        #   implement “half least squares loss” and “half poisson deviance” 
        #   to simplify the computation of the gradient. 
        # Furthermore, “poisson” loss internally uses a log-link and requires y >= 0
    
