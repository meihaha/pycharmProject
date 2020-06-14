# 基于运营商过往数据，采用决策树算法，对用户离网情况进行预测

import pandas as pd
from sklearn import model_selection
import graphviz
from sklearn import tree
import pydotplus
from  sklearn.tree import export_graphviz

data = pd.read_csv('churn.csv',encoding='utf-8',sep=',')
print('Data shape\n',data.shape)
print(data.head(5))

data =data.drop('Phone',axis=1)

# 编码转换，将yes和no转换为数值
from sklearn.preprocessing import  LabelEncoder
encoder = LabelEncoder()
data.iloc[:,0] = encoder.fit_transform((data.iloc[:,0]))
data.iloc[:,3] = encoder.fit_transform((data.iloc[:,3]))
data.iloc[:,4] = encoder.fit_transform((data.iloc[:,4]))
data.iloc[:,-1] = encoder.fit_transform((data.iloc[:,-1]))
print(data.head(5))

col_dicts = {}
cols =data.columns.values.tolist()
# [1:-1]去头去尾
X= data.loc[:,cols[1:-1]]
y =data[cols[-1]]

X_train,X_test,y_train,y_test = model_selection.train_test_split(X,y,test_size=0.3,random_state=0)
print(y_train.value_counts()/len(y_train))
print(y_test.value_counts()/len(y_test))

from sklearn.tree import DecisionTreeClassifier
tree_clf = DecisionTreeClassifier()
tree_clf.fit(X_train,y_train)

# 混淆矩阵
from sklearn.metrics import  classification_report
y_pred = tree_clf.predict(X_test)
target_names = ['class 0','class 1']
print(classification_report(y_test,y_pred,target_names=target_names))

# 调用gridSearchCV进行参数调优，调整的参数包括：树的最大深度，内部节点划分的最小样本数，叶子节点最小样本数
from sklearn.model_selection import GridSearchCV
model_to_set = DecisionTreeClassifier()
parameters = {'criterion':['entropy'],
              "max_depth":[3,4,5,6,7,8,9,10,11,12,13],
              "min_samples_split":[5,10,15,20,25,30,35,40,45,50],
              "min_samples_leaf":[5,10,15,20,25,30,35,40,45,50],
              }
model_tunning = GridSearchCV(model_to_set,param_grid=parameters,cv = 5)
model_tunning.fit(X_train,y_train)
print(model_tunning.best_score_)
print(model_tunning.best_params_)

tree_clf = DecisionTreeClassifier(max_depth=5,min_samples_leaf=5,min_samples_split=15)
tree_clf.fit(X_train,y_train)

y_pred = tree_clf.predict(X_test)
target_names = ['class 0','class 1']
print(classification_report(y_test,y_pred,target_names=target_names))

# 树的可视化
feature_names = X.columns[:]
class_names = ['0','1']

# 创建dot数据
dot_data = tree.export_graphviz(tree_clf,out_file=None,
                                feature_names = feature_names,
                                class_names = class_names,
                                filled=True,impurity= False)
graph = pydotplus.graph_from_dot_data(dot_data)
graph.write_png("tree.png")






