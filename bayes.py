import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss
from sklearn.naive_bayes import BernoulliNB
from sklearn.linear_model import LogisticRegression
import time

import warnings
warnings.filterwarnings('ignore')

train = pd.read_csv('train.csv',parse_dates=['Dates'])
test = pd.read_csv('test.csv',parse_dates=['Dates'])

# 数据预处理，犯罪进行编号，选择其中的星期几，小时和地区作为预测的影响因素，建立新的数据集
# 用label对不同的犯罪类型编号
leCrime = preprocessing.LabelEncoder()
crime = leCrime.fit_transform(train['Category'])

days = pd.get_dummies(train.DayOfWeek)
district = pd.get_dummies(train.PdDistrict)
hour = train.Dates.dt.hour
hour = pd.get_dummies(hour)
# 组合特征
trainData = pd.concat([hour,days,district],axis= 1)
trainData['crime'] = crime

# 测试数据做相同处理
days = pd.get_dummies(test.DayOfWeek)
district = pd.get_dummies(test.PdDistrict)
hour = test.Dates.dt.hour
hour = pd.get_dummies(hour)
testData = pd.concat([hour,days,district],axis= 1)


# 只取星期几和街区作为分类器输入特征
features = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday',
            'Sunday','BAYVIEW','CENTRAL','INGLESIDE','MISSION','NORTHERN','PARK'
            ,'RICHMOND','SOUTHERN','TARAVAL','TENDERLOIN']

# 分割训练集和测试集
training,validation = train_test_split(trainData,train_size = 0.6)

# 朴素贝叶斯建模，计算log_los
# time.time返回当前时间戳
# validation
model = BernoulliNB()
nbStart = time.time()
model.fit(training[features],training['crime'])
nbCostTime = time.time() - nbStart
predicted = np.array(model.predict_proba(validation[features]))
print("朴素贝叶斯建模耗时 %f 秒" % nbCostTime)
print("朴素贝叶斯log损失为 %f " % (log_loss(validation['crime'],predicted)))

model = LogisticRegression(C=0.1)
lrStrat = time.time()
model.fit(training[features],training['crime'])
lrCostTime = time.time() - lrStrat
predicted = np.array(model.predict_proba(validation[features]))
print("逻辑回归建模耗时 %f 秒" % lrCostTime)
print("逻辑回归log损失为 %f " % (log_loss(validation['crime'],predicted)))

# 加入犯罪时间点

features = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday',
            'Sunday','BAYVIEW','CENTRAL','INGLESIDE','MISSION','NORTHERN','PARK'
            ,'RICHMOND','SOUTHERN','TARAVAL','TENDERLOIN']
hourFea = [x for x in range(0,24)]
features = features + hourFea
training,validation = train_test_split(trainData,train_size = 0.6)
model = BernoulliNB()
nbStart = time.time()
model.fit(training[features],training['crime'])
nbCostTime = time.time() - nbStart
predicted = np.array(model.predict_proba(validation[features]))
print("朴素贝叶斯建模耗时 %f 秒" % nbCostTime)
print("朴素贝叶斯log损失为 %f " % (log_loss(validation['crime'],predicted)))

model = LogisticRegression(C=0.1)
lrStrat = time.time()
model.fit(training[features],training['crime'])
lrCostTime = time.time() - lrStrat
predicted = np.array(model.predict_proba(validation[features]))
print("逻辑回归建模耗时 %f 秒" % lrCostTime)
print("逻辑回归log损失为 %f " % (log_loss(validation['crime'],predicted)))