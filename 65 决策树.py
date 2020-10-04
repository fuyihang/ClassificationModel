#-*- coding: utf-8 -*-

########  本文件实现决策树模型，包括
# Part1、二分类决策树（违约）
# Part2、超参优化
# Part3、多分类决策树（鸢尾花）
# Part4、实战：通信套餐精准推荐
# Part5、决策回归树(CART回归树)
###################################

import pandas as pd
import numpy as np

from common import displayClassifierMetrics
from common import displayROCurve
from common import saveDTree

######################################################################
########  Part1、二分类决策树
######################################################################
# 目标变量只有二种取值

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

for i, col in enumerate(catCols):
    print('\n{}:{}'.format(col, enc.categories_[i]))

# 3）合并
X = pd.concat([dfCats, df[intCols]], axis=1)
cols = X.columns.tolist()

# 3、训练模型
from sklearn.tree import DecisionTreeClassifier

mdl = DecisionTreeClassifier(criterion = 'entropy')
mdl.fit(X, y )

# 4、评估模型
y_pred = mdl.predict(X)
displayClassifierMetrics(y, y_pred, mdl.classes_, poslabel)

y_prob = mdl.predict_proba(X)
displayROCurve(y, y_prob, mdl.classes_, '决策树')

# 5、筛选重要特征-手工
# 显示特征重要性
sr = pd.Series(mdl.feature_importances_, index = cols, name='决策树')
sr.sort_values(ascending=False, inplace=True)
sr.plot(kind='bar', title='特征重要性')

# 找出重要性累积超过85%的自变量
sr = sr.cumsum()
cond = sr < 0.85
k = len(sr[cond]) + 1
cols = sr.index[:k].tolist()

X = X[cols]

# 再次训练模型
# max_depth=4
mdl = DecisionTreeClassifier(
    criterion = 'entropy',
    min_samples_split=100,
    min_samples_leaf=50,
    max_depth=5
    )
mdl.fit(X, y )

# 评估模型
y_pred = mdl.predict(X)
displayClassifierMetrics(y, y_pred, mdl.classes_, poslabel)

y_prob = mdl.predict_proba(X)
displayROCurve(y, y_prob, mdl.classes_, '决策树')

# 6、应用模型
# 1）保存决策树（PDF文件）
saveDTree(mdl, cols, mdl.classes_)

# from sklearn.tree import plot_tree
# plot_tree(mdl)

# 2）保存模型（略）
# 3）加载模型（略）
# 4）预测（略）


# # 需要安装conda install python-graphviz
# from sklearn.tree import export_graphviz
# import graphviz

# data = export_graphviz(mdl, out_file=None, 
#         # max_depth = 5,       #树的最大深度
#         feature_names=cols,     #指定特征名称
#         class_names= mdl.classes_,  #指定类别取值
#         label='all',            #在哪些节点显示impurity指标
#         filled=True,            #显示子节点的主要类别
#         leaves_parallel=False,  #在询问画叶子节点
#         impurity = False,        #显示混杂度
#         node_ids = True,        #显示节点ID号
#         proportion = True,      #显示类别所占百分比，而不是数量
#         rotate=False,           #树的方向。True:从左到右,False:从上到下
#         rounded=True,           #画圆角而不是长方形
#         special_characters=False, #忽略特殊字符
#         precision=2 )           #数值小数位
# graph = graphviz.Source(data)
# graph.render("out")    #保存文件名,系统会自动加上后缀名.pdf


######################################################################
########  Part2、超参数最优配置
######################################################################
# 超参优化
# 常见的超参优化
#   criterion='gini','entropy'
# 预剪枝参数
#   max_depth=None, 一般特征少或样本少时不用管，解决过拟合问题
#   max_leaf_nodes=None, 最大叶子节点数
#   min_samples_leaf=1,  叶子节点最少样本数
#   min_samples_split=2, 节点划分所需的最小样本数
# 后剪枝参数
#   ccp_alpha=0.0,

# 设置参数矩阵
# 第一种方式，列表,多种组合。
# params = [{'criterion': ['entropy'], 
#             'min_impurity_decrease': np.linspace(0, 1, 100), 
#             'max_depth': np.arange(2,10)},
#         {'criterion': ['gini'], 
#             'min_impurity_decrease': np.linspace(0, 0.2, 100), 
#             'max_depth': np.arange(2,10)}]
# 第二种方式，字典。
params = {
            'criterion':['gini','entropy'],
            'max_depth': range(3,14,2), 
            'min_samples_split':range(20,210,20), 
            'min_samples_leaf':range(10,60,10)
        }
mdl = DecisionTreeClassifier(splitter='best',
        random_state=10)

from sklearn.model_selection import GridSearchCV

grid = GridSearchCV(estimator=mdl, param_grid=params)
grid.fit(X, y)


print("最优得分：",grid.best_score_)
print("最优超参：",grid.best_params_)
bestParams = grid.best_params_

# 预测
mdl = grid.best_estimator_

y_pred = mdl.predict(X)
displayClassifierMetrics(y, y_pred, mdl.classes_, poslabel)

y_prob = mdl.predict_proba(X)
displayROCurve(y, y_prob, mdl.classes_, '决策树')

# 评估（略）
# 显示特征重要性（略）
# 保存模型（略）
# 保存决策树（略）

######################################################################
########  Part3、多分类决策树（鸢尾花）
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

# 3.训练模型
from sklearn.tree import DecisionTreeClassifier
mdl = DecisionTreeClassifier(criterion = 'entropy', max_depth=4)
mdl.fit(X, y )


# 4.评估模型
y_pred = mdl.predict(X)
displayClassifierMetrics(y, y_pred, mdl.classes_)

y_prob = mdl.predict_proba(X)
displayROCurve(y, y_prob, mdl.classes_, '决策树')

# 显示特征重要性
sr = pd.Series(mdl.feature_importances_, index = cols, name='决策树')
sr.sort_values(ascending=False, inplace=True)
sr.plot(kind='bar', title='特征重要性')

# 5.超参优化（略）
# 6.应用模型（略）


######################################################################
########  Part4、实战：套餐类型精准推荐模型
######################################################################
# 目标变量取多个值，# 套餐类型建模

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
mdl = DecisionTreeClassifier(criterion = 'entropy')
mdl.fit(X, y )  #训练模型

# 4、评估模型
y_pred = mdl.predict(X)
displayClassifierMetrics(y, y_pred, mdl.classes_)

y_prob = mdl.predict_proba(X)
displayROCurve(y, y_prob, mdl.classes_, '多分类决策树')

# 显示特征重要性
sr = pd.Series(mdl.feature_importances_, index=cols, name='特征重要性')
sr.sort_values(ascending=False, inplace=True)
sr.plot(kind='bar', title=sr.name)

# 5、优化模型
# 1）筛选重要特征
sr = sr.cumsum()

cond = sr > 0.85
k = len(sr[cond]) + 1
cols = sr.index[:k].tolist()

X = X[cols]

# 2）超参优化
params = {
            'criterion':['gini','entropy'],
            'max_depth': range(len(cols),3*len(cols),2), 
            'min_samples_split':range(50,210,20), 
            'min_samples_leaf':range(10,60,10)
        }
mdl = DecisionTreeClassifier(splitter='best',
        random_state=10)

grid = GridSearchCV(estimator=mdl, param_grid=params)
grid.fit(X, y)

print("最优得分：",grid.best_score_)
print("最优超参：",grid.best_params_)
bestParams = grid.best_params_

# 3）优化后剪枝参数
params = {'ccp_alpha': np.linspace(0.0, 1.0, 100)}
mdl = DecisionTreeClassifier(**bestParams,
                splitter='best',random_state=10)
grid = GridSearchCV(estimator=mdl, param_grid=params)
grid.fit(X, y)

print("最优得分：",grid.best_score_)
print("最优超参：",grid.best_params_)
bestParams.update(grid.best_params_)

# 4）最优模型评估
mdl = grid.best_estimator_
y_pred = mdl.predict(X)
displayClassifierMetrics(y, y_pred, mdl.classes_)

y_prob = mdl.predict_proba(X)
displayROCurve(y, y_prob, mdl.classes_, '多分类决策树')

# 6、应用模型（略）
