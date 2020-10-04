#-*- coding: utf-8 -*- 

########  本文实现随机森林模型示例，包括：
# Part1、分类：RandomForestClassifier
# Part2、超参优化RandomForestClassifier
# Part3、实战：通信套餐精准推荐
# Part4、回归：RandomForestRegressor
###################################

import pandas as pd
import numpy as np

from common import displayClassifierMetrics
from common import displayROCurve

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor

# from sklearn.ensemble import RandomTreesEmbedding
from sklearn.model_selection import GridSearchCV

######################################################################
########  Part1、随机森林RandomForestClassifier
######################################################################

# 1、读取数据
filename = '分类预测.xls'
sheet = 'Titanic'
df = pd.read_excel(filename, sheet)
# print(df.columns.tolist())

# 2、特征工程
# 1）特征标识/选取
catCols = ['Sex','Embarked']
intCols = ['Pclass', 'Age', 'SibSp', 'Parch', 'Fare']
cols = catCols + intCols

target = 'Survived'
y = df[target]
poslabel = 1

# 2）数据预处理等
def prepare(data):
    # 对于年龄字段发生缺失，我们用所有年龄的均值替代
    data['Age'].fillna(data['Age'].mean(), inplace=True)
    data['Fare'].fillna(data['Fare'].mean(), inplace=True)
    
    # 性别男： 用0替代, 女：用1代替
    data['Sex'][data['Sex']=='male'] = 1
    data['Sex'][data['Sex']=='female'] = 0
    
    data['Embarked'].fillna('S', inplace=True)
    data['Embarked'][data['Embarked']=='S'] = 0
    data['Embarked'][data['Embarked']=='C'] = 1
    data['Embarked'][data['Embarked']=='Q'] = 2

prepare(df)

X = df[cols]

# 3、建立随机森林模型
from sklearn.ensemble import RandomForestClassifier

mdl = RandomForestClassifier(
    max_features = 0.8,
    n_estimators=51,
    min_samples_split=10,
    min_samples_leaf=5,
    oob_score=True,
    random_state=10)
mdl.fit(X, y)

# 4、评估模型
print('袋外得分=', mdl.oob_score_)  # 相当于泛化准确度

y_pred = mdl.predict(X)
displayClassifierMetrics(y, y_pred, mdl.classes_, poslabel)

y_prob = mdl.predict_proba(X)
displayROCurve(y, y_prob, mdl.classes_, '随机森林')

# 显示特征重要性
sr = pd.Series(mdl.feature_importances_, 
        index = cols, name='特征重要性')
sr.sort_values(ascending=False, inplace=True)
sr.plot(kind='bar', title=sr.name)

# 5.超参优化（略）
# 6.应用模型（略）

######################################################################
########  Part2、RF分类的超参优化
######################################################################
# 超参分两部分：
# 1）RF框架参数：
# n_estimators

# 2）决策树超参
# criterion
# max_depth, max_features,
# min_samples_split, min_samples_leaf, max_leaf_nodes,


bestParams = {}
# 1、调优参数：n_estimators
params = {'n_estimators':range(10,100,10)}
mdl = RandomForestClassifier(
        min_samples_split=10,
        min_samples_leaf=5,
        max_depth=8,
        max_features=0.8,
        oob_score=True,
        random_state=10)
grid = GridSearchCV(estimator=mdl, param_grid=params)
grid.fit(X, y)

# 评估模型
print('最优得分=',grid.best_score_)
mdl = grid.best_estimator_
print('袋外得分=', mdl.oob_score_)

# 保存最优参数
print('最优参数：', grid.best_params_)
bestParams.update(grid.best_params_)

# 2、调优预剪枝参数
params = {'max_depth':range(3,14,2), 
        'min_samples_split':range(50,201,20),
        'min_samples_leaf':range(10,60,10)}
mdl.set_params(**bestParams)    #不改变其它默认参数，只更新已经调优好的参数

grid = GridSearchCV(estimator=mdl, param_grid=params)
grid.fit(X, y)

# 评估
print('最优得分=',grid.best_score_)
mdl = grid.best_estimator_
print('袋外得分=', mdl.oob_score_)

# 保存最优参数
print('最优参数：', grid.best_params_)
bestParams.update(grid.best_params_)


# 3、调优参数：max_features （最优肯定是1.0呀）
params = {'max_features':np.linspace(0.5, 1.0,6)}

mdl.set_params(**bestParams)

grid = GridSearchCV(estimator=mdl, param_grid=params)
grid.fit(X, y)

# 评估
print('最优得分=',grid.best_score_)
mdl = grid.best_estimator_
print('袋外得分=', mdl.oob_score_)

# 保存最优参数
print('最优参数：', grid.best_params_)
bestParams.update(grid.best_params_)


###################################################
# 其实，上面进行三次调优，而且存在重复的代码，完全可以for循环合并起来
list_Params = [
        (1,{'n_estimators':range(10,100,10)}),  # 第一次
        (2,{'max_depth':range(3,14,2),          #第二次
            'min_samples_split':range(50,201,20),
            'min_samples_leaf':range(10,60,10)}),
        (3,{'max_features':np.linspace(0.5, 1.0,6)})    #第三次
        ]
bestParams = {}
# 使用初始的值进行构造 
mdl = RandomForestClassifier(
        min_samples_split=10,
        min_samples_leaf=5,
        max_depth=8,
        max_features=0.8,
        oob_score=True,
        random_state=10)
for i, params in list_Params:
    print("第{}调优".format(i))
    mdl.set_params(**bestParams)
    grid = GridSearchCV(estimator=mdl, param_grid=params, scoring='roc_auc', cv=5)
    grid.fit(X, y)

    # 评估
    print('最优得分=',grid.best_score_)
    mdl = grid.best_estimator_
    print('袋外得分=', mdl.oob_score_)

    # 保存最优参数
    print('最优参数：', grid.best_params_)
    bestParams.update(grid.best_params_)
print('最优超参：\n', bestParams)


######################################################################
########  Part3、实战：通信套餐精准推荐
######################################################################


# 1、读取数据
filename = 'Telephone.csv'
df = pd.read_csv(filename, encoding='gbk')
print(df.columns.tolist())

# 2、特征工程
# 1）特征标识/选取
catCols = ['居住地', '婚姻状况', '教育水平', '性别']
intCols = ['年龄', '收入', '家庭人数', '开通月数']
cols = catCols + intCols

target = '套餐类型'
y = df[target]



######################################################################
########  Part4、随机森林RandomForestRegressor
######################################################################

# 1、读取数据
filename = 'Telephone.csv'
df = pd.read_csv(filename, encoding='gbk')
# print(df.columns.tolist())

# 2、特征工程
# 1）特征标识/选取
catCols = ['居住地', '婚姻状况', '教育水平', '性别']
intCols = ['年龄', '收入', '家庭人数', '开通月数']
cols = catCols + intCols

target = '消费金额'
y = df[target]


# RandomForestClassifier
    # RandomForestClassifier(bootstrap=True, ccp_alpha=0.0, class_weight=None,
                    #    criterion='gini', max_depth=None, max_features='auto',
                    #    max_leaf_nodes=None, max_samples=None,
                    #    min_impurity_decrease=0.0, min_impurity_split=None,
                    #    min_samples_leaf=1, min_samples_split=2,
                    #    min_weight_fraction_leaf=0.0, n_estimators=100,
                    #    n_jobs=None, oob_score=False, random_state=None,
                    #    verbose=0, warm_start=False)

# 请参考“集成算法.py”中的Bagging集成
# 1）RF框架参数
    # 与BaggingClassifier基本相同，除了：
    # 1）没有的参数
        # base_estimator : 默认为决策树。
        # bootstrap_features：默认是无放回抽取特征
# 2）决策树相关参数，请参考决策树的参数

# 重要属性（和BaggingClassifer完全相同）
