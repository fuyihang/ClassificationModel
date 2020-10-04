#-*- coding: utf-8 -*-

########  本文件实现LightGBM模型，包括
# Part1、LGBMClassifier
# Part2、超参优化：GridSearchCV
# Part3、LGBMRegressor
###################################
# 安装:pip install lightgbm

import pandas as pd
import numpy as np

from common import displayClassifierMetrics
from common import displayRegressionMetrics
from common import displayROCurve

from sklearn.model_selection import GridSearchCV

######################################################################
########  Part1、LGBMClassifier
######################################################################

# 1、读取数据
filename = 'Telephone.csv'
df = pd.read_csv(filename, encoding='gbk')
# print(df.columns.tolist())

# 2、数据处理
# 1)特征标识
intCols = ['年龄','收入','家庭人数','开通月数']
catCols = ['居住地','婚姻状况','教育水平','性别']

target = '套餐类型'
y = df[target]

# 2）类别变量-->数字化
from sklearn.preprocessing import OrdinalEncoder

enc = OrdinalEncoder(dtype='int')
X_ = enc.fit_transform(df[catCols])

dfCats = pd.DataFrame(X_, columns=catCols)

for i, col in enumerate(catCols):
    print('\n{}:{}'.format(col ,enc.categories_[i]))

# 3）合并
X = pd.concat([dfCats, df[intCols]], axis=1)
cols = X.columns.tolist()


# 3、模型训练
from lightgbm import LGBMClassifier, LGBMRegressor

gbm = LGBMClassifier(
            objective='multiclass', 
            num_leaves=31, 
            learning_rate=0.05, 
            n_estimators=20)
gbm.fit(X, y)


# 模型评估
y_pred = gbm.predict(X)
displayClassifierMetrics(y, y_pred, gbm.classes_)

y_prob = gbm.predict_proba(X)
displayROCurve(y, y_prob, gbm.classes_, 'LightGBM')

# 显示特征重要性


######################################################################
########  Part2、超参优化
######################################################################

bestParams = {}
lt_params = [
    {'n_estimators':range(50,150,10)},

    {'max_depth':range(3,14),
        'min_child_weight':range(1,6)},

    # {'gamma': [i/10.0 for i in range(0,6)]},

    # {'subsample': [i/10.0 for i in range(6,10)],
    #     'colsample_bytree':[i/10.0 for i in range(6,10)]},

    # {'reg_alpha': [1e-2, 0.1, 1, 2, 5, 10],
    #     'reg_lambda': [0.01,0.1, 1, 2, 5, 10]},
    
    # {'learning_rate':np.linspace(0.01, 1.0, 50)}
]

for params in lt_params:
    grid = GridSearchCV(estimator = gbm, 
                       param_grid = params)
    grid.fit(X,y)

    bestParams.update(grid.best_params_)
    gbm.set_params(**bestParams)

print('最优参数：\n', bestParams)
print('score=', grid.best_score_)
mdl = grid.best_estimator_

y_pred = mdl.predict(X, num_iteration=mdl.best_iteration_)
displayClassifierMetrics(y, y_pred, grid.classes_)

y_prob = mdl.predict_proba(X, num_iteration=mdl.best_iteration_)
displayROCurve(y, y_prob, grid.classes_)

# 显示特征重要性


# lightgbm.LGBMClassifier
    #       (boosting_type='gbdt', class_weight=None, colsample_bytree=1.0,
    #       importance_type='split', learning_rate=0.1, max_depth=-1,
    #       min_child_samples=20, min_child_weight=0.001, min_split_gain=0.0,
    #       n_estimators=100, n_jobs=-1, num_leaves=31, objective=None,
    #       random_state=None, reg_alpha=0.0, reg_lambda=0.0, silent=True,
    #       subsample=1.0, subsample_for_bin=200000, subsample_freq=0)

    # 重要参数
        # boosting_type=‘gbdt','dart','goss','rf' 提升类型
        # n_estimators=100
        # learning_rate: 0.1,  # 学习速率，较小

    # 决策树
        # objective='None','regression','binary','multiclass','lambdarank' 学习的目标函数
        # class_weight='None','balanced', or dict
        # reg_alpha=0, L1正则项
        # reg_lambda=0, L2正则项

    # 抽样参数
        # subsample=1.0
        # subsample_freq=0
        # subsample_for_bin=200000, 分箱个数, 较大更准确，但会导致速度变慢
        # colsample_bytree=1.0

    # 剪枝参数
        # max_depth=-1，典型值[5,10]
        # num_leaves: 31,  # 叶子节点数，一般设置为小于2^max_depth。过大，会导致过拟合
        # importance_type='split','gain'
        # min_split_gain=0,  最小切分的信息增益值
        # min_child_weight=1e-3, 
        # min_child_samples=20,

    # 其它参数
        # random_state=None
        # n_jobs=-1,
        # silent=True    

# 属性
    # best_iteration_
    # best_score_
    # booster_
    # classes_
    # n_classes_
    # evals_result_
    # n_features_
    # feature_importances_
    # feature_name_
    # objective_
