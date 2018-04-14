from sklearn import datasets
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier  # WHAT IS sklearn.neighbors? why from 'skelearn.neighbors import *' also works
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier

#PRESS SHIFT 2 TIMES TO SEARCH EVERYWHERE

numbers = datasets.load_digits()

# for i in range(len(numbers.))
# print(numbers.data.reshape(1797,8,8))  # a = numbers.data has (1797,64) to convert into images need to do--> a.reshape(1797,8,8) and this becomes the same as numbers.images

numbers['images'] = numbers.data.reshape(1797,8,8) # create new feature called images

# print (numbers.images) # can either use numbers['images'] or numbers.images

# fig, axes = plt.subplots(5,5)
# for i, ax in enumerate(axes.flat): # WHAT DOES AXES.FLAT DO?
#     ax.imshow(numbers.images[i], cmap ='binary', interpolation = 'nearest')
# plt.show() # if put plt.show inside the for loop then it replaces always the same



# a = np.random.normal(0,1,(5,5)) # TRAIN YOUR DATA AND THEN GIVE IT RANDOM NUMBERS AND ASK WHICH NUMBER IT IS
# plt.imshow(a)
# plt.show()

numbers_train_X = numbers.data[:-10]
numbers_train_Y = numbers.target[:-10]
numbers_test_X = numbers.data[-10:] # use [-10:] , [-10:0] does not work
numbers_test_Y = numbers.target[-10:]


knn = KNeighborsClassifier()
knn.fit(numbers_train_X,numbers_train_Y)
predict = knn.predict(numbers_test_X)
print(predict)
print(numbers_test_Y)

lin = LinearRegression()
lin.fit(numbers_train_X,numbers_train_Y)
predict = lin.predict(numbers_test_X)
print(predict)
print(numbers_test_Y )

lor = LogisticRegression()
lor.fit(numbers_train_X,numbers_train_Y)
predict = lor.predict(numbers_test_X)
print(predict)
print(numbers_test_Y)

sgd = SGDClassifier(random_state=42)
sgd.fit(numbers_train_X,numbers_train_Y)
predict = sgd.predict(numbers_test_X)
print(predict)
print(numbers_test_Y)