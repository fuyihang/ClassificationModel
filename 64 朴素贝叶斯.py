#-*- coding: utf-8 -*-

########  本文件实现三类朴素贝叶斯模型，包括
# Part1、GaussianNB(高斯朴素贝叶斯)
# Part2、MultinomialNB(多项式朴素贝叶斯)
# Part3、BernoulliNB(伯努利朴素贝叶斯)
# Part4、ComplementNB（补充朴素贝叶斯）
###################################

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from common import displayClassifierMetrics
from common import displayROCurve

######################################################################
########  Part1、高斯贝叶斯GaussianNB
######################################################################

# 1、读入数据
filename = '分类预测.xls'
sheet = '贷款违约'
df = pd.read_excel(filename, sheet)
# print(df.columns.tolist())

# 2、数据处理

# 1）特征标识
intCols = ['年龄', '工龄', '地址', '收入', '负债率', '信用卡负债', '其他负债']
catCols = ['学历']

target = '违约'
y = df[target]
poslabel = '是'

# 2）类别变量数字化
from sklearn.preprocessing import OrdinalEncoder

enc = OrdinalEncoder(dtype='int')
X_ = enc.fit_transform(df[catCols])

# 映射关系
for i, col in enumerate(catCols):
    print('\n变量名称：', col)
    print('数值顺序',enc.categories_[i])

dfCats = pd.DataFrame(X_, df.index, catCols)

# 3)合并
X = pd.concat([df[intCols], dfCats], axis=1)
cols = X.columns.tolist()

# 3.训练模型
from sklearn.naive_bayes import GaussianNB

mdl = GaussianNB()
mdl.fit(X, y)

# 4.评估模型
y_pred = mdl.predict(X)
displayClassifierMetrics(y, y_pred, mdl.classes_, poslabel)

y_prob = mdl.predict_proba(X)
displayROCurve(y, y_prob, mdl.classes_, '朴素贝叶斯')

# 其余略


