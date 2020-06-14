import numpy as np
import pandas as pd
from pandas import Series,DataFrame
import matplotlib.pyplot as plt
import sklearn.datasets as datasets

# 导入显现回归机器算法模型
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from  sklearn.svm import SVR
from  sklearn.tree import DecisionTreeRegressor
from  sklearn.model_selection import train_test_split
from  sklearn.preprocessing import StandardScaler

# 查看关键数据集
boston = datasets.load_boston()
boston_df = pd.DataFrame(boston.data,columns=boston.feature_names)
boston_df['房价值'] = boston.target
# print(boston_df.head(10))

train = boston.data
target = boston.target
X_train,x_test,y_train,y_true = train_test_split(train,target,test_size = 0.2)
X_train_scaler = StandardScaler().fit_transform(X_train)
x_test_scaler = StandardScaler().fit_transform(x_test)
print('训练数据的均值',np.mean(X_train))
print('训练数据的方差',np.var(X_train))
print('测试数据的均值',np.mean(x_test))
print('测试数据的方差',np.var(x_test))
print('标准化后训练数据的均值',np.mean(X_train_scaler))
print('标准化后训练数据的方差',np.var(X_train_scaler))
print('标准化后测试数据的均值',np.mean(x_test_scaler))
print('标准化后测试数据的方差',np.var(x_test_scaler))

# 创建学习模型
knn = KNeighborsRegressor()
linear = LinearRegression()
ridge = Ridge()
lasso = Lasso()
decision = DecisionTreeRegressor()
svr = SVR()

# 开始做训练
knn.fit(X_train,y_train)
linear.fit(X_train,y_train)
ridge.fit(X_train,y_train)
lasso.fit(X_train,y_train)
decision.fit(X_train,y_train)
svr.fit(X_train,y_train)

y_predict_knn = knn.predict(x_test)
y_predict_lin = linear.predict(x_test)
y_predict_ridge = ridge.predict(x_test)
y_predict_lasso = lasso.predict(x_test)
y_predict_decision = decision.predict(x_test)
y_predict_svr = svr.predict(x_test)

# R2评估，越接近1越好
from sklearn.metrics import r2_score
knn_scrore = r2_score(y_true,y_predict_knn)
linear_scrore = r2_score(y_true,y_predict_lin)
ridge_scrore = r2_score(y_true,y_predict_ridge)
lasso_scrore = r2_score(y_true,y_predict_lasso)
decision_scrore = r2_score(y_true,y_predict_decision)
svr_scrore = r2_score(y_true,y_predict_svr)
print('KNN score:  ',knn_scrore)
print('Linear score:  ',linear_scrore)
print('Ridge score:  ',ridge_scrore)
print('Lasso score:  ',lasso_scrore)
print('Decision score:  ',decision_scrore)
print('SVR score:  ',svr_scrore)

#KNN
plt.plot(y_true,label='true')
plt.plot(y_predict_knn,label='knn')
plt.legend()
plt.show()

#Linear
plt.plot(y_true,label='true')
plt.plot(y_predict_lin,label='linear')
plt.legend()
plt.show()

#Ridge
plt.plot(y_true,label='true')
plt.plot(y_predict_ridge,label='ridge')
plt.legend( )
plt.show()

#lasso
plt.plot(y_true,label='true')
plt.plot(y_predict_lasso,label='lasso')
plt.legend()
plt.show()


#decision
plt.plot(y_true,label='true')
plt.plot(y_predict_decision,label='decision')
plt.legend()
plt.show()

#SVR
plt.plot(y_true,label='true')
plt.plot(y_predict_svr,label='svr')
plt.legend()
plt.show()