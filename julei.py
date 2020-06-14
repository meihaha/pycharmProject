import pandas as pd
import numpy as np
import matplotlib.pylab as plt

X = pd.read_csv("telecom.csv",encoding='utf-8')

from sklearn import preprocessing
X_scaled = preprocessing.scale(X)
X_scaled[0:5]

from sklearn.decomposition import PCA
pca = PCA(n_components=3)
X_pca = pca.fit_transform(X_scaled)
X_pca_frame = pd.DataFrame(X_pca,columns=['pca_1','pca_2','pca_3'])
X_pca_frame.head(10)

# 利用Kmeans進行聚類
from sklearn.cluster import KMeans
est = KMeans(n_clusters=10)
est.fit(X_pca)

kmeans_clustering_labels = pd.DataFrame(est.labels_,columns=['cluster'])
X_pca_frame = pd.concat([X_pca_frame,kmeans_clustering_labels],axis=1)
X_pca_frame.head()

# 对不同的K值进行进行测算，挑选出最优值
from mpl_toolkits.mplot3d import  Axes3D
from  sklearn import metrics

# kmeans实例化，将其设置为K= range(2,14)
d = {}
fig_reduced_data = plt.figure(figsize=(12,12))
for k in range(2,14):
    est = KMeans(n_clusters=k,random_state=111)
#     作用到降维数据上
    y_pred = est.fit_predict(X_pca)
    calinski_harabaz_score = metrics.calinski_harabaz_score(X_pca_frame,y_pred)
    d.update({k:calinski_harabaz_score})
    print('calinski_harabaz_score with k={0} is {1}'.format(k,calinski_harabaz_score))
    ax = plt.subplot(4,3,k-1,projection='3d')
    ax.scatter(X_pca_frame.pca_1,X_pca_frame.pca_2,X_pca_frame.pca_3,c = y_pred)
    ax.set_xlabel('pca1')
    ax.set_ylabel('pca2')
    ax.set_zlabel('pca3')
plt.show()
# 寻找最优的K值
x = []
y = []
for k,score in d.items():
    x.append(k)
    y.append(score)

plt.plot(x,y)
plt.xlabel('k value')
plt.ylabel('calinski_harabaz_score')

# 样本筛选
X.index = X_pca_frame.index
X_full  = pd.concat([X,X_pca_frame],axis=1)
X_full.head()

#去除异常样本点
grouped = X_full.groupby('cluster')
result_data = pd.DataFrame()
for name,group in grouped:
    print('Group:{0},sample_before:{1}'.format(name,group['pca_1'].count()))
    desp = group[['pca_1','pca_2','pca_3']].describe()
    for att in ['pca_1','pca_2','pca_3']:
        lower25 = desp.loc['25%',att]
        upper75 = desp.loc['75%',att]
        IQR = upper75 - lower25
        min_value = lower25 - 1.5*IQR
        max_value = upper75 + 1.5*IQR
#         删除噪声
        group = group[(group[att]>min_value)&(group[att]<max_value)]
    result_data = pd.concat([result_data,group],axis=0)
#     每组去除异常值的个数
    print('Group:{0},Samples after:{1}'.format(name,group['pca_1'].count()))
print('Remain sample:',result_data['pca_1'].count())

# 原始数据降维后的可视化，非去除异常点
from  mpl_toolkits.mplot3d import Axes3D
fig_reduced_data = plt.figure()
ax_reduced_data = plt.subplot(111,projection = '3d')
ax_reduced_data.scatter(X_pca_frame.pca_1.values,X_pca_frame.pca_2.values,X_pca_frame.pca_3.values)
ax_reduced_data.set_xlabel('Compent_1')
ax_reduced_data.set_ylabel('Compent_2')
ax_reduced_data.set_zlabel('Compent_3')
plt.show()
# 聚类后将不同簇的数据可视化
# 注意颜色不能写错，否则对应有问题
cluster_2_color = {0:'red',1:'green',2:'blue',3:'yellow',4:'cyan',5:
    'black',6:'magenta',7:'#fff0f5',8:'#ffdab9',9:'#ffa500'}
# 运用map方法将数字对应到对应的字符串颜色
colors_clustered_data = X_pca_frame.cluster.map(cluster_2_color)
fig_scatted_data = plt.figure()
ax_clustered_data = plt.subplot(111,projection = '3d')
ax_clustered_data.scatter(X_pca_frame.pca_1.values,
                          X_pca_frame.pca_2.values,
                          X_pca_frame.pca_3.values,
                          c=colors_clustered_data)
ax_clustered_data.set_xlabel('Compent_1')
ax_clustered_data.set_ylabel('Compent_2')
ax_clustered_data.set_zlabel('Compent_3')
plt.show()


# 筛选后的数据进行可视化分析
colors_filtered_data = result_data.cluster.map(cluster_2_color)
fig = plt.figure()
ax= plt.subplot(111,projection = '3d')
ax.scatter(result_data.pca_1.values,
           result_data.pca_2.values,
           result_data.pca_3.values,
           c = colors_filtered_data)
ax.set_xlabel('Compent_1')
ax.set_ylabel('Compent_2')
ax.set_zlabel('Compent_3')
plt.show()

# 用户画像构建,通过几个字段分析不同用户的特点
# 查看每个月话费等特征情况
monthly_Fare = result_data.groupby('cluster').describe().loc[:,u'每月话费']
print(monthly_Fare)
monthly_Fare[['mean','std']].plot(kind = 'bar',rot = 0,legend = True,title = '每月话费')
plt.show()

access_time = result_data.groupby('cluster').describe().loc[:,u'入网时间']
print(access_time)
access_time[['mean','std']].plot(kind = 'bar',rot = 0,legend = True,title = '入网时间')
plt.show()
arrearage= result_data.groupby('cluster').describe().loc[:,u'欠费金额']
print(arrearage)
arrearage[['mean','std']].plot(kind = 'bar',rot = 0,legend = True,title = '欠费金额')
plt.show()

new_column = ['Access_time',u'套餐价格',u'每月流量','monthly_Fare',u'每月通话时长','Arrearage',u'欠费月份数',
    u'pca_1',u'pca_2',u'pca_3',u'cluster']
result_data.columns = new_column
result_data.groupby('cluster')[['monthly_Fare','Access_time','Arrearage']].mean().plot(kind = 'bar')
plt.show()

