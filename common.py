#-*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 特征选择：利用相关性
import statsmodels.formula.api as smf
import statsmodels.stats.anova as smsa
def featureSelection(df, intCols, catCols, target, threshold=0.05):
    '''\
        实现特征选择，利用协方差分析
    '''
    # 仅做主效应检验
    formula = target + ' ~ ' + \
                '+'.join(intCols) + '+' + \
                '+'.join(catCols)                

    module = smf.ols(formula, df).fit()
    dfRet = smsa.anova_lm(module)

    # 取显著因子项
    cond = dfRet['PR(>F)'] < threshold
    cols = dfRet[cond].index.tolist()
    # print(cols)

    # 筛选变量
    for col in intCols:
        if col not in cols:
            intCols.remove(col)
    for col in catCols:
        if col not in cols:
            catCols.remove(col)
    return intCols, catCols

def intFeatureSelection(df, intCols, target, threshold=0.3):
    '''\
        实现数值型变量的特征选择，利用相关系数矩阵
    '''
    dfcorr = df[intCols+[target]].corr(method='spearman')

    cond = np.abs(dfcorr[target]) > threshold
    cols = dfcorr[cond].index.tolist()
    cols.remove(target)
    return cols

# 显示回归评估指标
from sklearn import metrics
def displayRegressionMetrics(y_true, y_pred, adjVal=None):
    '''
    \n功能：计算回归的各种评估指标。
    \n参数：y_true:真实值
         y_pred:预测值
         adjVal:输入的shape参数(n,p)，其中n是样本量，p是特征数
            默认None表示是一元回归；
    \n返回：各种指标，字典形式
    '''
    # 评估指标：R^2/adjR^2, MAPE, MAE，RMSE
    mts = {}
    #一元回归，计算R^2；
    mts['R2'] = metrics.r2_score(y_true, y_pred)
    # 多元回归，计算调整R^2
    if (adjVal != None) and (adjVal[1] > 1):
        n, p = adjVal
        mts['adjR2']  = 1-(1-mts['R2'])*(n-1)/(n-p-1)

    mts['MAPE'] = (abs((y_pred-y_true)/y_true)).mean()
    mts['MAE'] = metrics.mean_absolute_error(y_true, y_pred)
    MSE = metrics.mean_squared_error(y_true, y_pred)
    mts['RMSE'] = np.sqrt(MSE)
    
    # 格式化，保留小数点后4位
    for k,v in mts.items():
        mts[k] = np.round(v, 4)
    
    # 特别处理,注意变成了字符串
    mts['MAPE'] = '{0:.2%}'.format(mts['MAPE']) 

    # print('回归模型评估指标：\n', mts)
    
    return mts


# 画学习曲线
from sklearn.model_selection import learning_curve
def plot_learning_curve(estimator, X, y, cv=None, scoring = None):
    train_sizes, train_scores, test_scores = learning_curve(
            estimator, X, y, cv = cv,scoring=scoring, 
            train_sizes=np.linspace(0.1, 1.0, 50))

    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    plt.figure()
    plt.title("学习网线")
    plt.xlabel('训练样本数量')
    plt.ylabel(scoring)
    plt.grid()

    skipNum = 3
    plt.fill_between(train_sizes[skipNum:], 
                train_scores_mean[skipNum:] - train_scores_std[skipNum:],
                train_scores_mean[skipNum:] + train_scores_std[skipNum:], 
                alpha=0.1, color="b")
    plt.fill_between(train_sizes[skipNum:], 
                test_scores_mean[skipNum:] - test_scores_std[skipNum:],
                test_scores_mean[skipNum:] + test_scores_std[skipNum:], 
                alpha=0.1, color="g")

    plt.plot(train_sizes[skipNum:], train_scores_mean[skipNum:], 
            'o-', color="b", label="训练集")
    plt.plot(train_sizes[skipNum:], test_scores_mean[skipNum:], 
            'o-', color="r", label="测试集")

    plt.legend(loc="best")
    plt.show()


# 自定义类：实现特征选择、类别变量哑变量化、数值变量标准化
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder

import statsmodels.formula.api as smf
import statsmodels.stats.anova as sms

class MyFeaturePreprocessing(object):
    def __init__(self,normalize=False):
        super().__init__()
        self.cols = []
        self.catCols = []
        self.intCols = []
        self.target = ''

        self.pthreshold = 0.05
        self.normalize = normalize

        self.encCat = OneHotEncoder(drop='first', sparse=False)
        self.encInt = StandardScaler()

    def fit(self, X, y=None):
        """主要实现如下功能：\n
        1）负责筛选出显著影响的因素
        2）对类别型变量进行哑变量转换
        3）对数值型变量进行标准化处理
        """
        self.cols = []
        self.catCols = []
        self.intCols = []
        self.target = ''

        df = pd.concat([X, y], axis=1)
        # print(df.dtypes)
        self.target = y.name
        cols = X.columns.tolist()

        # 1)自动识别变量类型
        for col in cols:
            if np.issubdtype(df.dtypes[col], np.number):
                self.intCols.append(col)
            else:
                self.catCols.append(col)
        # print(self.intCols,"\n", self.catCols,"\n", self.target)

        # 2)找出显著相关的变量
        formula = '{} ~ {}'.format(
                        self.target,
                        '+'.join(cols))
        module = smf.ols(formula, df).fit()
        dfanova = sms.anova_lm(module)
        # print(dfanova)
        
        cond = dfanova['PR(>F)'] < self.pthreshold
        cols = dfanova[cond].index.tolist()
        # print('显著影响因素：\n',cols)

        for col in self.intCols:
            if col not in cols:
                self.intCols.remove(col)
        for col in self.catCols:
            if col not in cols:
                self.catCols.remove(col)

        # 3）类别变量--哑变量化
        if self.catCols != []:
            # self.encCat = OneHotEncoder(drop='first', sparse=False)
            self.encCat.fit(df[self.catCols], y)

        # 4）数值变量--标准化
        if self.normalize and self.intCols != []:
            # self.encInt = StandardScaler()
            self.encInt.fit(df[self.intCols], y)

        return self

    def transform(self, X, copy=None):
        df = pd.DataFrame(X)

        # 1）类别变量转换
        X_ = self.encCat.transform(df[self.catCols])

        cols = []
        for i in range(len(self.encCat.categories_)):
            cols.extend( self.encCat.categories_[i][1:].tolist() )
        
        dfCats = pd.DataFrame(X_, index=df.index, columns=cols)

        # 2）数值变量转换
        if self.normalize:
            X_ = self.encInt.transform(df[self.intCols])
            dfInts = pd.DataFrame(X_, index=df.index,columns=self.intCols)
        else:
            dfInts = df[self.intCols]
        # 3）合并
        dfRet = pd.concat([dfCats, dfInts], axis=1)
        self.cols = dfRet.columns.tolist()

        return dfRet

    def fit_transform(self, X, y=None):
        self.fit(X, y)
        dfRet = self.transform(X)
        return dfRet

# 显著分类模型混淆矩阵和评估指标
def displayClassifierMetrics(y_true:np.ndarray, y_pred:np.ndarray,labels=[0,1],pos_label=1):
    '''
    功能：  计算分类模型的混淆矩阵，以及评估指标.
    参数：  y_true:真实值数组
            y_pred:预测值数组
            labels:标签列表
            pos_label:正类标签，此值必须出现在labels中，二分类时使用。
    返回：混淆矩阵，评估指标字典。即(dfMatrix, mts)
    '''
    # 混淆矩阵,合计，转化
    matrix = metrics.confusion_matrix(y_true, y_pred)

    dfMatrix = pd.DataFrame(matrix, columns=labels, index=labels)
    dfMatrix['合计'] = dfMatrix.sum(axis=1)
    dfMatrix = dfMatrix.append(pd.Series(dfMatrix.sum(axis=0), name='合计'))
    # print('混淆矩阵：\n', dfMatrix)

    # 计算评估指标
    mts = {}
    mts['Accuracy'] = metrics.accuracy_score(y_true, y_pred)   #正确率
    
    if len(labels) > 2 :   #多分类时
        avgs = ['micro','macro']   #暂不考虑'weighted','samples'
    else:
        avgs = ['binary']
    for avg in avgs:
        if avg == 'binary':
            suffix = ''
        else:
            suffix = '_' + avg
        key = 'Precision' + suffix
        mts[key] = metrics.precision_score(y_true, y_pred,labels=labels,pos_label=pos_label,average=avg)
        key = 'Recall' + suffix
        mts[key] =  metrics.recall_score(y_true, y_pred,labels=labels,pos_label=pos_label,average=avg)
        key = 'F1' + suffix
        mts[key] =  metrics.f1_score(y_true, y_pred, labels=labels, pos_label=pos_label,average=avg)

    # 格式化一下，保留4位小数
    for k,v in mts.items():
        mts[k] = np.round(v, 4)
    # print(mts)

    return dfMatrix, mts

# 显示ROC曲线和AUC值
def displayROCurve(y_true:np.ndarray, y_prob:np.ndarray, labels=[0,1], title='ROC曲线'):

    # plt.figure()
    nPlot = len(labels)  #子图个数

    for pos, label in enumerate(labels):
        # 计算相关指标
        fpr, tpr, _ = metrics.roc_curve(y_true, y_prob[:,pos], pos_label=label)
        # pos_label正类的标签，默认为None(即1)
        auc = metrics.auc(fpr, tpr)     #AUC,利用fpr, tpr计算
        # auc = metrics.roc_auc_score(y_true, y_prob[:,pos])

        plt.subplot(1, nPlot, pos+1)
        plt.plot(fpr, tpr, label='{}'.format(label))
        plt.plot([0,1], [0,1], linestyle='--', color='k', label='random chance')
        plt.text(0, 1, "AUC={0:.6f}".format(auc))
        plt.legend(loc='lower right')

        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')

    plt.suptitle(title)
    plt.show()

    return

# 保存决策树为PDF文件
# 1.先在系统中安装graphviz
# MacOS系数：brew install graphviz
# Windows系统：在https://graphviz.org/download/ 下载graphviz-2.38.msi
    # 在用户变量Path路径，添加C:\Program Files(x86)\Graphviz2.38
    # 在系统变量Path路径，添加C:\Program Files(x86)\Graphviz2.38\bin
# 2.安装conda install python-graphviz 
    # 或者pip install graphviz
# 3.导入
from sklearn.tree import export_graphviz
import graphviz
def saveDTree(mdl, cols, ylabels, filename='out'):
    '''\
        保存决策树为PDF文件
    '''
    data = export_graphviz(mdl, out_file=None, 
            # max_depth = 5,        #树的最大深度
            feature_names=cols,     #指定特征名称
            class_names= ylabels,   #指定类别标签值
            label='all',            #在哪些节点显示impurity指标
            filled=True,            #显示子节点的主要类别
            leaves_parallel=False,  #在底端画叶子节点
            impurity = False,        #是否显示熵值或基尼系数的值
            node_ids = True,        #显示节点ID号
            proportion = False,      #显示类别所占百分比(True)，而不是数量(False)
            rotate=False,           #树的方向。True:从左到右,False:从上到下
            rounded=True,           #画圆角而不是长方形
            special_characters=False, #忽略特殊字符
            precision=2 )           #数值小数位
    graph = graphviz.Source(data)
    graph.render(filename)    #保存文件名,系统会自动加上后缀名.pdf


# Scoring取值范围，可以使用sorted(sklearn.metrics.SCORERS.keys())获得
# # Regression
    # ‘explained_variance’            metrics.explained_variance_score
    # ‘max_error’                     metrics.max_error
    # ‘neg_mean_absolute_error’       metrics.mean_absolute_error
    # ‘neg_mean_squared_error’        metrics.mean_squared_error
    # ‘neg_root_mean_squared_error’   metrics.mean_squared_error
    # ‘neg_mean_squared_log_error’    metrics.mean_squared_log_error
    # ‘neg_median_absolute_error’     metrics.median_absolute_error
    # ‘r2’                            metrics.r2_score
    # ‘neg_mean_poisson_deviance’     metrics.mean_poisson_deviance
    # ‘neg_mean_gamma_deviance’       metrics.mean_gamma_deviance

# # Classification
    # ‘accuracy’              metrics.accuracy_score
    # ‘balanced_accuracy’     metrics.balanced_accuracy_score
    # ‘average_precision’     metrics.average_precision_score
    # ‘neg_brier_score’       metrics.brier_score_loss
    # ‘f1’                    metrics.f1_score
    # ‘f1_micro’              metrics.f1_score        #micro-averaged
    # ‘f1_macro’              metrics.f1_score        #macro-averaged
    # ‘f1_weighted’           metrics.f1_score        #weighted average
    # ‘f1_samples’            metrics.f1_score        #by multilabel sample
    # ‘neg_log_loss’          metrics.log_loss        #requires predict_proba support
    # ‘precision’ etc.        metrics.precision_score #suffixes apply as with ‘f1’
    # ‘recall’ etc.           metrics.recall_score    #suffixes apply as with ‘f1’
    # ‘jaccard’ etc.          metrics.jaccard_score   #suffixes apply as with ‘f1’
    # ‘roc_auc’               metrics.roc_auc_score   
    # ‘roc_auc_ovr’           metrics.roc_auc_score
    # ‘roc_auc_ovo’           metrics.roc_auc_score
    # ‘roc_auc_ovr_weighted’  metrics.roc_auc_score
    # ‘roc_auc_ovo_weighted’  metrics.roc_auc_score

# # Clustering
    # ‘adjusted_mutual_info_score’    metrics.adjusted_mutual_info_score
    # ‘adjusted_rand_score’           metrics.adjusted_rand_score
    # ‘completeness_score’            metrics.completeness_score
    # ‘fowlkes_mallows_score’         metrics.fowlkes_mallows_score
    # ‘homogeneity_score’             metrics.homogeneity_score
    # ‘mutual_info_score’             metrics.mutual_info_score
    # ‘normalized_mutual_info_score’  metrics.normalized_mutual_info_score
    # ‘v_measure_score’               metrics.v_measure_score

