#-*- coding: utf-8 -*-

########  本文件实现K近邻分类模型，包括
# Part1、最近邻类：NearestNeighbors
# Part2、最近邻分类：KNeighborsClassifier
# Part3、

##################################
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from common import displayClassifierMetrics
from common import displayROCurve

# 通用类
from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import DistanceMetric    #距离度量类
# 回归
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neighbors import RadiusNeighborsRegressor
# 分类
from sklearn.neighbors import KNeighborsClassifier  #k个近邻
from sklearn.neighbors import RadiusNeighborsClassifier #半径r内的近邻
# 算法类
from sklearn.neighbors import BallTree
from sklearn.neighbors import KDTree
# 有监督降维
from sklearn.neighbors import NeighborhoodComponentsAnalysis

# 
from sklearn.neighbors import KNeighborsTransformer
from sklearn.neighbors import RadiusNeighborsTransformer


############################
######## Part1、最近邻类：NearestNeighbors
############################
# 该类用来计算最近邻的距离、最近邻的样本点

# 1、读取数据
filename = '分类预测.xls'
sheet = '约会评估'
df = pd.read_excel(filename, sheet)
# print(df.columns.tolist())
# 网站约会评估数据集，一个客户对约会对象的评估

# 2、数据预处理
cols = ['飞行公里数','玩游戏时间占比','吃冰淇淋公升数']
target = '是否喜欢'
y = df[target]

# 标准化
from sklearn.preprocessing import StandardScaler
enc = StandardScaler()
X = enc.fit_transform(df[cols])

# 3、计算最近邻
from sklearn.neighbors import NearestNeighbors

K = 5
# 算法有auto, ball_tree, kd_tree, brute。当为auto时，算法会尝试确定最佳方法
mdl = NearestNeighbors(n_neighbors=K, 
                        algorithm='ball_tree',
                        metric='minkowski')
mdl.fit(X)

# 返回每个样本的最近邻距离和索引
distances, indices = mdl.kneighbors(X)
# distances返回最近的几个点的距离，其中第一个为0
# indices返回最近的几个点的索引，其中第一个为本身的索引号


############################
######## Part2、最近邻分类：KNeighborsClassifier
############################

# 3.训练模型
from sklearn.neighbors import KNeighborsClassifier

mdl = KNeighborsClassifier(n_neighbors=K, weights='uniform', 
            metric='minkowski')
mdl.fit(X,y)

# 5.超参优化（略）
from sklearn.model_selection import GridSearchCV
params = {'n_neighbors':range(1,10)}
mdl = KNeighborsClassifier()
grid = GridSearchCV(mdl, param_grid=params)
grid.fit(X, y)

print('最优参数:', grid.best_params_)
print('最优得分:', grid.best_score_)

mdl = grid.best_estimator_

# 6.评估模型
y_pred = mdl.predict(X)
displayClassifierMetrics(y, y_pred, mdl.classes_)

y_prob = mdl.predict_proba(X)
displayROCurve(y, y_prob, mdl.classes_)

# 相关类
    # KNeighborsClassifier(n_neighbors=5,weights=’uniform’,algorithm=’auto’,
    #               leaf_size=30,p=2,metric=’minkowski’,metric_params=None,n_jobs=1,*kwargs)
    # n_neighbors: int, 可选参数(默认为 5)
    # weights（权重）: str or callable(自定义类型), 可选参数(默认为 ‘uniform’)
    # 用于预测的权重函数。可选参数如下:
        # - ‘uniform’ : 统一的权重. 在每一个邻居区域里的点的权重都是一样的。
        # - ‘distance’ : 权重点等于他们距离的倒数。使用此函数，更近的邻居对于所预测的点的影响更大。
        # - [callable] : 一个用户自定义的方法，此方法接收一个距离的数组，然后返回一个相同形状并且包含权重的数组。
    # algorithm（算法）: {‘auto’, ‘ball_tree’, ‘kd_tree’, ‘brute’}, 可选参数（默认为 'auto'）
    # 计算最近邻居用的算法：
        # - ‘ball_tree’ 使用算法[BallTree]，适合高维数据集
        # - ‘kd_tree’ 使用算法[KDTree]，适合低维数据集
        # - ‘brute’ 使用暴力搜索计算，适用小数据集
        # - ‘auto’ 会基于传入自动处理
        # 注意 : 如果传入fit方法的输入是稀疏的，将会重载参数设置，直接使用暴力搜索。

    # leaf_size（叶子数量）: int, 可选参数(默认为 30)
        # 传入BallTree或者KDTree算法的叶子数量。
        # 此参数会影响构建、查询BallTree或者KDTree的速度，以及存储BallTree或者KDTree所需要的内存大小。 
        # 当小于此值时，切换成brute暴力计算算法
    # p: integer, 可选参数(默认为 2)
        # 用于Minkowski metric[闵可夫斯基空间]的超参数。
        # p = 1, 相当于使用[曼哈顿距离] (l1)，
        # p = 2, 相当于使用[欧几里得距离](l2)
        # 对于任何 p ，使用的是[闵可夫斯基空间]
    #  metric（矩阵）: string or callable, 默认为 ‘minkowski’
    # 用于树的距离矩阵。默认为[minkowski闵可夫斯基空间]，如果和p=2一块使用相当于使用标准欧几里得矩阵. 
    # 所有可用的矩阵列表请查询 DistanceMetric 的文档。
    # metric_params（矩阵参数）: dict, 可选参数(默认为 None)。给矩阵方法使用的其他的关键词参数。
    # n_jobs: int, 可选参数(默认为 1)。用于搜索邻居的，可并行运行的任务数量。
    # 如果为-1, 任务数量设置为CPU核的数量。
