import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

data = pd.read_csv("churn.csv",encoding='utf-8',sep=',')
# 导入数据集
dataset = pd.read_csv

X = dataset[:,[2,3]].values
y = dataset[:,4].values

# 确定每一次划分的集合都一样，random_state = 0进行确定
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_true =train_test_split(X,y,test_size = 0.25,random_state= 0 )


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state=0)
classifier.fit(X_train,y_train)

y_predict = classifier.predict(X_test)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_true,y_predict)
print(cm)
print('Accuracy of LR Classfier: ',classifier.score(X_test,y_true))

from matplotlib.colors import ListedColormap
X_set ,y_set = X_train,y_train
X1,X2 = np.meshgrid(np.arange(start=X_set[:,0].min()-1,stop=X_set[:,0].max+1,step=0.01),
                    np.arange(start=X_set[:,1].min()-1,stop=X_set[:,1].max+1,step=0.01))
print(X1,X2)
