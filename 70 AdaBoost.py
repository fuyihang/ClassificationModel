#-*- coding: utf-8 -*-

########  本文件实现Boosting集成模型：AdaBoost自适应提升树
# Part1、二分类：AdaBoostClassifier
# Part2、超参优化
# Part3、回归：AdaBoostRegressor
###################################

import pandas as pd
import numpy as np

from common import displayClassifierMetrics
from common import displayROCurve


from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import AdaBoostRegressor
from sklearn.tree import DecisionTreeClassifier

from sklearn.model_selection import GridSearchCV


######################################################################
########  Part1、二分类：AdaBoostClassifier
######################################################################

# 1、读取数据
#################生成数据集################
from sklearn.datasets import make_gaussian_quantiles
import matplotlib.pyplot as plt

x1, y1 = make_gaussian_quantiles(cov=2.0, n_samples=200, n_features=2, n_classes=2, random_state=1)
x2, y2 = make_gaussian_quantiles(mean=(3,3), cov=1.5, n_samples=300, n_features=2, n_classes=2, random_state=1)

# 2、数据处理（略）

X = np.concatenate((x1, x2))
y = np.concatenate((y1, 1-y2))
cols = ['X1', 'X2']

plt.scatter(X[:,0], X[:,1], marker='o', c=y)
plt.show()


# 3、训练模型
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

mdl = DecisionTreeClassifier(max_depth=1)
        # min_samples_split=20,
        # min_samples_leaf=5)
clf = AdaBoostClassifier(
    base_estimator= mdl,
    algorithm='SAMME',
    n_estimators=200,
    # learning_rate=0.7,
    # random_state=10
)
clf.fit(X, y)

# 3、评估模型
print('score=', clf.score(X,y))

y_pred = clf.predict(X)
displayClassifierMetrics(y, y_pred, clf.classes_)

y_prob = clf.predict_proba(X)
displayROCurve(y, y_prob, clf.classes_, 'AdaBoost')

# 1）显示特征重要性
sr = pd.Series(clf.feature_importances_, index = cols, name='特征重要性')
sr.sort_values(ascending=False, inplace=True)
sr.plot(kind='bar',title=sr.name)

# 2）其余基类信息
print('类别取值个数：', clf.n_classes_)
print('类别标签取值：', clf.classes_)
for i, est in enumerate(clf.estimators_):
    print('第{}个基学习器：'.format(i))
    # mdl = est   #可以保存起来，后续使用
    print('     权重:', np.round(clf.estimator_weights_[i], 4))
    print('     分类错误率：',np.round(clf.estimator_errors_[i],4))


######################################################################
########  Part2、超参优化：AdaBoostClassifier
######################################################################
# 1）Adaboost的框架参数
# n_estimators，调参时和learning_rate同时考虑。
# learning_rate：每个弱分类器的权重缩减系数v。
#       强学习器的迭代公式为fk(x) = fk-1(x) + v*ak*Gk(x).
#       v的取值范围为(0,1]。v较小，则需要的弱分类器更多，拟合效果会下降。
#       通常和n_estimators一起调参。
# algorithm：'SAMME','SAMME.R'（仅分类）
#   
# loss：    'linear','square','exponential'（仅回归）

# 2）基学习器参数（略，参考决策超参）

# from sklearn.datasets import make_gaussian_quantiles
# import matplotlib.pyplot as plt

# x1, y1 = make_gaussian_quantiles(cov=2.0, n_samples=200, n_features=2, n_classes=2, random_state=1)
# x2, y2 = make_gaussian_quantiles(mean=(3,3), cov=1.5, n_samples=300, n_features=2, n_classes=2, random_state=1)

# # 2、数据处理（略）

# X = np.concatenate((x1, x2))
# y = np.concatenate((y1, 1-y2))

############################
######## 先对框架超参调优
############################
params = {'n_estimators':range(10,20,10),
        'learning_rate':np.linspace(0.1,1,10)}

mdl = DecisionTreeClassifier(max_depth=1)
clf = AdaBoostClassifier(
    base_estimator= mdl,
    algorithm='SAMME',
    n_estimators=200,
    learning_rate=0.8,
    random_state=10)

grid = GridSearchCV(estimator=clf, param_grid=params)
grid.fit(X, y)

# 评估
print('最优得分=',grid.best_score_)
mdl = grid.best_estimator_

# 保存最优参数
print('最优参数：', grid.best_params_)
bestclfParams = grid.best_params_

############################
######## 再对基学习器超参调优
############################
# 可惜目前只能手工遍历
from sklearn.model_selection import cross_val_score
mdl = DecisionTreeClassifier()
clf = AdaBoostClassifier(
    base_estimator= mdl,
    algorithm='SAMME',
    n_estimators=200,
    learning_rate=0.8,
    random_state=10)
clf.set_params(**bestclfParams)

# 定义调参步骤
dtcParams = {'max_depth':range(3,14,2),        #第一次
            'min_samples_split':range(50,201,20),
            'min_samples_leaf':range(10,60,10)}
# dtcParams_2 = {'max_features':np.linspace(0.5, 1.0,6)}    #第二次

for var_name, var_vals in dtcParams.items():
    best_scores = []
    best_Params = []
    for val in var_vals:
        dv = {var_name:val}
        best_Params.append(dv)
        mdl.set_params(**dv)
        scores = cross_val_score(clf, X, y)
        best_scores.append(scores.mean())
    
    # 评估
    idx = np.argmax(best_scores)
    print('最优得分=',best_scores[idx])
    print('最优参数=', best_Params[idx])


# sklearn.ensemble.AdaBoostClassifier(
    #           base_estimator=None, 
    #           n_estimators=50, 
    #           learning_rate=1.0, 
    #           algorithm=’SAMME.R’, 
    #           random_state=None)
    # 算法参数
        # base_estimator=None, 基学习器
        # n_estimators=50, 基学习器个数
        # learning_rate=1.0,学习率，取值范围为(0,1]
        # 越小，则分类效果会下降，这意味着需要越多的迭代次数/分类器个数
        # algorithm='SAMME.R', 算法（仅回归）。取'SAMME','SAMME.R'
        #       SAMME算法使用对样本集分类效果作为弱分类器权重；
        #       而SAMME.R使用了样本分类预测概率大小作为弱分类器权重，迭代比SAMME快。
        #       如果选择算法为SAMME.R，还需要基类支持概率预测。
    # 其它参数
        # random_state=None

    # 重要属性
        # base_estimators_ :基学习器类型
        # estimators_:list of classifiers
        # classes_ :ndarray of shape (n_classes,)
        # n_classes_:int
        # estimator_weights_ :ndarray of floats学习器权重
        # estimator_errors_: ndarray of floats学习器的分类错误率
        # feature_importance: ndarray of shape (n_features,) 特征重要性

# 超参优化
    # n_estimators和learning_rate同时一起调参。
    # 算法一般默认使用'SAMME.R'，调参的意义不大
    # 然后，再对基学习器调参

# sklearn.ensemble.AdaBoostRegressor
    #       (base_estimator=None, 
    #       learning_rate=1.0, 
    #       loss='linear',
    #       n_estimators=50, 
    #       random_state=None)
    # 参数基本同上AdaBoostClassifier。
    # 区别在于：
    #   1）没有algorithm参数，转为为adaboost.R2算法
    #   2) 新增参数loss='linear'，表示指定误差率公式
        # loss='linear'。{‘linear’, ‘square’, ‘exponential’}
        #       若为linear，则eki=|yi-Gk(xi)|/Ek;
        #       若为square,则eki=(yi-Gk(xi))^2/Ek^2
        #       若为exponential，则eki=1-exp( (-yi+Gk(xi))/Ek )
        #       Ek为训练集上的最大误差Ek = max|yi-Gk(xi)| ,i=1,...,m
# 超参优化
    # n_estimators和learning_rate同时一起调参。
    # 再对loss调参
    # 然后，再对基学习器调参