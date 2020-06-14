# 预测现网具有单卡转合约倾向的合同
# 解决小样本，非线性，以及高纬识别问题有很大优势

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
# %matplotlib inline

from  sklearn import datasets
from  sklearn import svm,metrics
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV,RandomizedSearchCV
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

X_train,X_test,y_train,y_test = train_test_split(X_normaliazed,y,test_size=0.2,random_state=112)
print('The shape of X_train is {0}'.format(X_train.shape))
print('The shape of X_test is {0}'.format(X_test.shape))

plt.scatter(X_train[:,0],X_train[:,1],c = y_train)
plt.show()
# 训练简单模型
linear_clf = svm.LinearSVC()
linear_clf.fit(X_train,y_train)
y_pred = linear_clf.predict(X_test)

score = metrics.accuracy_score(y_test,y_pred)
print('The accuracy of the model is {0}'.format(score))

print(metrics.confusion_matrix(y_test,y_pred))

# 超参调节

nbStart = time.time()
C_range = np.logspace(-5,5,5)
gamma_shape = np.logspace(-9,2,10)

clf = svm.SVC(kernel='rbf',cache_size=1000,random_state=117)
param_grid = {'C':C_range,'gamma':gamma_shape}
# gridSearch作用在训练集上
grid = GridSearchCV(clf,param_grid = param_grid,scoring='accuracy',cv = 5)
grid.fit(X_train,y_train)

nbCostTime = time.time() - nbStart
print("gridsearch耗时 %f 秒" % nbCostTime)
print(grid.best_score_)
print(grid.best_params_)
print(grid.best_estimator_)

# 训练参数调优后的模型
nbStart= time.time()
clf_best = svm.SVC(kernel='rbf',C= grid.best_params_['C'],gamma=grid.best_params_['gamma'],probability=True)
clf_best.fit(X_train,y_train)
y2_pred = clf_best.predict(X_test)

nbCostTime = time.time() - nbStart
print("参数调优后的耗时 %f 秒" % nbCostTime)
# 预测结果评估
accuracy = metrics.accuracy_score(y_test,y2_pred)

print('The accuracy of the model is {0}'.format(accuracy))

print(metrics.confusion_matrix(y_test,y2_pred))

# 模型效果评估
y2_pred_prob = clf_best.predict_proba(X_test)[:,1]
# 获取roc曲线
fpr,tpr,thresholds = metrics.roc_curve(y_test,y2_pred_prob)
plt.plot(fpr,tpr)
plt.xlim([0.0,1.0])
plt.ylim([0.0,1.0])
plt.title('Roc curve for diabets classifier')
plt.xlabel('False positive rate(1-Specificity)')
plt.ylabel('True positive rate(Sensitivity)')
plt.grid(True)
