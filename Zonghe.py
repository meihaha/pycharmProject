# 本作业基于位置数据对海上目标进行智能识别和作业行为分析，
# 要求学员通过分析渔船北斗设备位置数据，得出该船的生产作业行为，
# 具体判断出是拖网作业、围网作业还是流刺网作业。
# 同时，希望选手通过数据可视分析，挖掘更多海洋通信导航设备的应用价值。

import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt

data = pd.read_csv('ocean_ship_data.csv',encoding='utf-8',sep=',')
data.describe()

from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
data.iloc[:,-1] = encoder.fit_transform(data.iloc[:,-1])
target = data.iloc[:,-1]
x = data.iloc[:,[0,1]]

X_train,x_test,y_train,y_test = train_test_split(x,target,test_size=0.3,random_state=0)

tree_clf = DecisionTreeClassifier()
tree_clf.fit(X_train,y_train)

y_pred = tree_clf.predict(x_test)
target_names = ['class0','class1','class2']
print(classification_report(y_test,y_pred,target_names=target_names))
plt.scatter(x.iloc[:,0],x.iloc[:,1],c=target)
plt.show()


