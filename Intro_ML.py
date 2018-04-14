import numpy as np
from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier

iris = datasets.load_iris()
iris_X = iris.data   # use iris_X.shape --> SHAPE shows the dimensions. it's a property of numpy arrays
iris_Y = iris.target

np.random.seed(0)
indices = np.random.permutation(len(iris_X))
iris_X_train = iris_X[indices[:-10]] # use index -10 as [:-10] to avoid having to specify the length of the array and just select all data from 0 except the last 10
iris_Y_train = iris_Y[indices[:-10]] # WHAT IS indices?b
iris_X_test = iris_X[indices[-10:]]
iris_Y_test = iris_Y[indices[-10:]]

knn = KNeighborsClassifier() # typing knn in hte console shows you hte details of the classifier or regressor
knn.fit(iris_X_train,iris_Y_train)
print(knn.predict(iris_X_test))
print(iris_Y_test)

#PRESS SHIFT 2 TIMES TO SEARCH EVERYWHERE