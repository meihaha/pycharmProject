# 预测现网具有单卡转合约倾向的合同

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
# %matplotlib inline

from  sklearn import datasets
from  sklearn import svm,metrics
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV,RandomizedSearchCV,cross_val_score
from  sklearn.preprocessing import OneHotEncoder

from pylab import mpl
# 指定默认字体
mpl.rcParams['font.sans-serif'] = ['FangSong']
# 解决保存图像是负号‘-’显示是方块的问题
mpl.rcParams['axes.unicode_minus'] = False

data = pd.read_csv('data_carrier_svm.csv',encoding = 'utf8')

cond = data['是否潜在合约用户'] ==1
data[cond]['主叫时长（分）'].hist(alpha = 0.5,label = '潜在合约用户')
data[~cond]['主叫时长（分）'].hist(color = 'r',alpha = 0.5,label = '非潜在合约用户')
plt.legend()
plt.show()

cond = data['是否潜在合约用户'] ==1
data[cond]['被叫时长（分）'].hist(alpha = 0.5,label = '潜在合约用户')
data[~cond]['被叫时长（分）'].hist(color = 'r',alpha = 0.5,label = '非潜在合约用户')
plt.legend()
plt.show()

# 筛选数量
grouped = data.groupby(['是否潜在合约用户','业务类型'])['用户标识'].count().unstack()
grouped.plot(kind = 'bar',alpha = 1,rot = 0)

y = data.loc[:,'是否潜在合约用户']
plt.scatter(data.loc[:,'主叫时长（分）'],data.loc[:,'免费流量'],c=y,alpha = 0.5)
plt.show()

# y = data['是否潜在合约用户']
# plt.scatter(data['主叫时长（分）'],data['免费流量'],c=y,alpha = 0.5)

# 数据预处理
# 这个用法得记住，切片
X = data.loc[:,'业务类型':'余额']
Y = data['是否潜在合约用户']
print('the shape of X is {0}'.format(X.shape))
print('the shape of Y is {0}'.format(Y.shape))

# 类别特征编码
service_df = pd.get_dummies(X['业务类型'])
X_enc = pd.concat([X,service_df],axis=1).drop('业务类型',axis = 1)

# 数字归一化
from sklearn.preprocessing import normalize
X_normaliazed = normalize(X_enc)

from sklearn.ensemble import GradientBoostingClassifier
X_train,X_test,y_train,y_test = train_test_split(X_normaliazed,y,test_size=0.2,random_state=112)
print('The shape of X_train is {0}'.format(X_train.shape))
print('The shape of X_test is {0}'.format(X_test.shape))

plt.scatter(X_train[:,0],X_train[:,1],c = y_train)
plt.show()
# 训练简单模型
# linear_clf = svm.LinearSVC()
# linear_clf.fit(X_train,y_train)
# y_pred = linear_clf.predict(X_test)
def modelfit(alg,X_train,y_train,performCV =True,printFeatureImportance= True,cv_folds = 5):
    alg.fit(X_train,y_train)
    train_predictions = alg.predict(X_train)
    train_preprob = alg.predict_proba(X_train)[:,1]
    cv_score = cross_val_score(alg,X_train,y_train,cv =  cv_folds,scoring='roc_auc')
    print('\nModel Report')
    print("Accuracy(Train):3.4%f"%metrics.accuracy_score(y_train.values,train_predictions))
    print('AUC score (Train):%f'%metrics.roc_auc_score(y_train,train_preprob))
    if performCV == True:
        print("CV score: Mean - %.7g| std - %.7g| Max - %.7g|Min - %.7g"%np.mean(cv_score),np.std(cv_score),np.max(cv_score),np.min((cv_score)))
    if printFeatureImportance:
        feat_imp = pd.Series(alg.feature_importance_,X_trian.columns.toList()).sort_values(ascending=True)
        feat_imp.plot(kind = 'barh',title = 'Feature Importance')
        plt.ylabel("Feature Importance Score")
        plt.xlabel('Relative importance')

clf0 = GradientBoostingClassifier(random_state= 10)
clf0.fit(X_train,y_train)
y_pred = clf0.predict(X_test)
score = metrics.accuracy_score(y_test,y_pred)
print('accuracy score of the model is :{0}'.format(score))

print(metrics.confusion_matrix(y_test,y_pred))
modelfit(clf0,X_train,y_train)

# GBDT调参数


