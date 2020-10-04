#-*- coding: utf-8 -*-

########  本文实现XGBoost模型示例，包括
# Part1、分类：XGBClassifier
# Part2、超参优化
# Part3、回归:
###################################
# 项目 https://github.com/dmlc/xgboost
# 官方文档# https://xgboost.readthedocs.io
# 安装xgboost模块：
# 1）brew install libomp
# 2）pip3 install xgboost

import pandas as pd
import numpy as np

from common import displayClassifierMetrics
from common import displayRegressionMetrics
from common import displayROCurve

from sklearn.model_selection import GridSearchCV

# from xgboost.sklearn import XGBClassifier
from xgboost import XGBClassifier, XGBRegressor
# from xgboost import XGBRFClassifier, XGBRFRegressor

######################################################################
########  Part1、XGBClassifier
######################################################################

# 1、读取数据
filename = 'Telephone.csv'
df = pd.read_csv(filename, encoding='gbk')
# print(df.columns.tolist())

# 2、特征工具
# 1）特征标识/筛选
intCols = ['年龄','收入', '家庭人数', '开通月数']
catCols = ['居住地', '婚姻状况', '教育水平', '性别']

target = '套餐类型'
y = df[target]

# 2）类别变量数字化
from sklearn.preprocessing import OrdinalEncoder

enc = OrdinalEncoder(dtype='int')
X_ = enc.fit_transform(df[catCols])

dfCats = pd.DataFrame(X_, columns=catCols)

# 3）合并
X = pd.concat([dfCats, df[intCols]], axis=1)
cols = X.columns.tolist()

# 3、训练模型
from xgboost import XGBClassifier

model = XGBClassifier(
        learning_rate=0.01,
        # n_estimators=3000,
        max_depth=4,
        min_child_weight=5,
        gamma=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=1,
        objective='binary:logistic',
        nthread=8,
        scale_pos_weight=1,
        seed=27
    )
model.fit(X, y)

# 模型评估
y_pred = model.predict(X)
displayClassifierMetrics(y, y_pred, model.classes_)

y_prob = model.predict_proba(X)
displayROCurve(y, y_prob, model.classes_, 'XGBoost')

######################################################################
########  Part2、超参优化
######################################################################
# 调优步骤：
# 1）学习率learning_rate [0.05, 0.3]
# 2）决策树超参:max_depth, min_child_weight,
# 3）节点分裂参数：gamma
# 4）抽样参数：subsample, colsample_bytree
# 5）正则化参数：reg_alpha, reg_lambda

# 默认的经验值
xgb = XGBClassifier(
    booster = 'gbtree',
    learning_rate = 0.1,
    n_estimators = 100,
    max_depth = 5,
    min_child_weight = 1,
    gamma = 0,
    subsample = 0.8,
    colsample_bytree = 0.8,
    objective = 'binary:logistic',
    nthread = 4,
    scale_pos_weight =None,
    seed = 27 )

bestParams = {}
lt_params = [
    {'n_estimators':range(50,150,10),
        'learning_rate': [0.01, 0.3]},

    {'max_depth':range(3,14),
        'min_child_weight':range(1,6)},

    {'gamma': [i/10.0 for i in range(0,6)]},

    {'subsample':[i/10.0 for i in range(6,10)],
        'colsample_bytree':[i/10.0 for i in range(6,10)]},

    {'reg_alpha':[1e-2, 0.1, 1, 2, 5, 10],
        'reg_lambda':[0.01,0.1, 1, 2, 5, 10]},
    
    {'learning_rate':np.linspace(0.01, 1.0,50)}
]

for params in lt_params:
    grid = GridSearchCV(estimator = xgb, 
                       param_grid = params)
    grid.fit(X,y)

    bestParams.update(grid.best_params_)
    # xgb.set_params(**bestParams)

print('最优参数：\n', bestParams)
print('score=', grid.best_score_)
mdl = grid.best_estimator_
y_pred = grid.predict(X)
displayClassifierMetrics(y, y_pred, grid.classes_)

y_prob = grid.predict_proba(X)
displayROCurve(y, y_prob, grid.classes_, 'XGBoost')

# 显示特征重要性(略)


# xgboost.XGBClassifier
    # (base_score=None, booster=None, colsample_bylevel=None,
    # colsample_bynode=None, colsample_bytree=None, gamma=None,
    # gpu_id=None, importance_type='gain', interaction_constraints=None,
    # learning_rate=None, max_delta_step=None, max_depth=None,
    # min_child_weight=None, missing=nan, monotone_constraints=None,
    # n_estimators=100, n_jobs=None, num_parallel_tree=None,
    # objective='binary:logistic', random_state=None, reg_alpha=None,
    # reg_lambda=None, scale_pos_weight=None, subsample=None,
    # tree_method=None, validate_parameters=None, verbosity=None)

    # XGBoost的参数，有三类：
    #     通用参数：宏观函数控制。
    #     Booster参数：控制每一步的booster(tree/regression)。
    #     学习目标参数：控制训练目标的表现。

    ##########框架参数
        # booster='gbtree'
            # 选择每次迭代的模型，有两种选择：
            # gbtree：基于树的模型
            # gbliner：线性模型.
            # dart: 表示采用dart booster
        # n_estimators=100

    ##########决策树参数
        # objective='binary:logistic'

        # num_parallel_tree 用于提升随机森林
        # monotone_constraints=None,
        # interaction_constraints=None
        # importance_type='gain', weight, cover,total_gain, total_cover, 指定特征重要类型
        # learning_rate=0.3, 越小，可提高模型的鲁棒性,典型值：0.01-0.2
        # reg_lambda = 1.0, L2正则项系数
        # reg_alpha=1.0 , L1正则化项系数
    # 分裂参数
        # max_depth=6, 典型值：3-10
        # gamma=0 在节点分裂时，只有分裂后损失函数的值下降了，才会分裂这个节点。
            # Gamma指定了节点分裂所需的最小损失函数下降值。
            # 这个参数的值越大，算法越保守。这个参数的值和损失函数息息相关，所以是需要调整的。
        # tree_method=’auto’ 指定了构建树的算法(近似算法)：
            # ‘auto’： 小数据集，用exact，大数据集用其它近似算法
            # ‘exact’： 使用exact greedy 算法分裂节点，适合小和中午的数据集
            # ‘approx’： 使用近似算法分裂节点
            # ‘hist’： 使用histogram 优化的近似算法分裂节点（比如使用了bin cacheing 优化）
            # ‘gpu_exact’： 基于GPU 的exact greedy 算法分裂节点
            # ‘gpu_hist’： 基于GPU 的histogram 算法分裂节点
        # min_child_weight=1.0, 最小叶子节点样本权重和。
            # 和GBM的 min_child_leaf 参数类似，但不完全一样。XGBoost的这个参数是最小样本权重的和，而GBM参数是最小样本总数。
            # 用于避免过拟合，越小越容易过拟合; 过高，会导致欠拟合。
        # max_delta_step=0
            # 这参数限制每棵树权重改变的最大步长。如果这个参数的值为0，那就意味着没有约束。
            # 如果它被赋予了某个正值，那么它会让这个算法更加保守。
            # 通常，这个参数不需要设置。但是当各类别的样本十分不平衡时，它对逻辑回归是很有帮助的。
    ##########抽样参数
        # subsample=1.0, 典型值:0.5-1.
            # 对样本随机抽样的比例。
            # 减小这个参数的值，算法会更加保守，避免过拟合。但是，如果这个值设置得过小，它可能会导致欠拟合。
        # colsample_bytree=1.0, 典型值：0.5-1，每一棵树的特征采样占比，类似RF的特征抽样。
        # colsample_bylevel=1.0, 每一级的特征采样占比。
        # colsample_bynode=1.0, 每次分裂的特征采样比例

        # scale_pos_weight = 1.0，在各类别样本十分不平衡时，把这个参数设定为一个正值，可以使算法更快收敛。

    ##########其它参数
        # n_jobs=None, 并行线程数
        # random_state=None
        # missing=np.nan, 缺失值表示
        # verbosity=0，输出信息级别，默认0(silent)-3(debug)
        # base_score=None, 所有样本的初始预测分。当迭代的数量足够大时，参数无影响
# ##########返回属性
    # 

