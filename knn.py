from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import matplotlib.pyplot as plt

def createDataset():
    dataset = np.array([[1,1],[1,1.5],[2,2.5],[2.5,3],[1.5,1],[3,2.5]])
    # print(dataset[:,0])
    label = ['A','A','B','B','A','B']
    plt.scatter(dataset[:,0],dataset[:,1])
    plt.show()
    return dataset,label
createDataset()

def classify(test,dataset,label,k):
    size = dataset.shape[0]
    diff = np.tile(test,(size,1))-dataset
    sqdiff = diff**2

#     对进行的向量求和
    dist = np.sum(sqdiff,axis=1)
    distIndex = np.argsort(dist)
    print(dist)
    print(distIndex)

    classCount = {}
    for i in range(k):
        votelable = label[distIndex[i]]
        classCount[votelable] = classCount.get(votelable,0)+1
    print(classCount)
    maxCount = 0
    for key,value in classCount.items():
        if maxCount < value:
            maxCount = value
            classlabel = key
    return classlabel

# if __name__ == "__main__":
#     dataset,label =  createDataset()
#     test = [1.75,1.75]
#     k = 4
#     result = classify(test,dataset,label,k)
#     print(result)


from sklearn.neighbors import  KNeighborsClassifier
import numpy as np
x = np.array([[1,1],[1,1.5],[2,2.5],[2.5,3],[1.5,1],[3,2.5]])
y= ['A','A','B','B','A','B']

# K是代表K个距离，
model = KNeighborsClassifier(n_neighbors=4,algorithm='ball_tree')
model.fit(x,y)

print(model.predict([[1.75,1.75]]))
print(model.predict_proba([[1.75,1.75]]))
print(model.score(x,y))

print('hello')




