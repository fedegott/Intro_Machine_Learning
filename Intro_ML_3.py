from sklearn import datasets
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier  # WHAT IS sklearn.neighbors? why from 'skelearn.neighbors import *' also works
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier

from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import cross_val_score

import seaborn as sns
sns.set_style('whitegrid') # FYI the “sns.set_style(‘whitegrid’)” just sets us up to use a nice pre set plot scheme, provided by the seaborn library

#PRESS SHIFT 2 TIMES TO SEARCH EVERYWHERE

numbers = datasets.load_digits()
print("The number of dimension of the data is {} and of the target is {}".format(numbers.data.shape, numbers.target.shape))
# plt.imshow(numbers.data)
# plt.show()

# for i in range(len(numbers.))
# print(numbers.data.reshape(1797,8,8))  # a = numbers.data has (1797,64) to convert into images need to do--> a.reshape(1797,8,8) and this becomes the same as numbers.images

# numbers['images'] = numbers.data.reshape(1797,8,8) # create new feature called images

# print (numbers.images) # can either use numbers['images'] or numbers.images

# fig, axes = plt.subplots(5,5)
# for i, ax in enumerate(axes.flat): # if you wanna iterary over the nd.array, you need to convert it into 1d. either .flat, .flatten() or .ravel() work fine
#     ax.imshow(numbers.images[i], cmap ='binary', interpolation = 'nearest') # axes contains all the 1797 ax
# plt.show() # if put plt.show inside the for loop then it replaces always the same


# plt.imshow(numbers.images[0])
# plt.show()


# a = np.random.normal(0,1,(5,5)) # TRAIN YOUR DATA AND THEN GIVE IT RANDOM NUMBERS AND ASK WHICH NUMBER IT IS
# plt.imshow(a)
# plt.show()

numbers_train_X = numbers.data[:-10]
numbers_train_Y = numbers.target[:-10]
# numbers_test_X = numbers.data[-10::-1] # 1788 instances from 1797
numbers_test_X = numbers.data[-10:] # 10 instances
numbers_test_Y = numbers.target[-10:]

# print(numbers.data[-10::-1].shape, numbers.data[-10:].shape)
predict=[]
predict_name = []

knn = KNeighborsClassifier()
knn.fit(numbers_train_X,numbers_train_Y)
predict_knn = knn.predict(numbers_test_X)
predict.append(predict_knn)
print(predict_knn)
print(numbers_test_Y)

predict_name.append(str(predict_knn))

lin = LinearRegression()
lin.fit(numbers_train_X,numbers_train_Y)
predict_LinR = lin.predict(numbers_test_X)
predict.append(predict_LinR)
print(predict)
print(numbers_test_Y )

# #
lor = LogisticRegression()
lor.fit(numbers_train_X,numbers_train_Y)
predict_logR = lor.predict(numbers_test_X)
predict.append(predict_logR)
print(predict)
print(numbers_test_Y)
# #
sgd = SGDClassifier(random_state=42)
sgd.fit(numbers_train_X,numbers_train_Y)
predict_SGDclass = sgd.predict(numbers_test_X)
predict.append(predict_SGDclass)
print(predict)
print(numbers_test_Y)

# fig, ax1 = plt.subplots(nrow = 3,ncols = 1)
# for i,_ in enumerate(predict):
numbers_test_Y_string = str(numbers_test_Y).strip('[]')
for i in range(4):

    plt.subplot(4,1,i+1)
    plt.plot(numbers_test_Y_string.split(),predict[i], marker= 'o', linestyle = '', c = 'b'  ,alpha = 0.5)

    # plt.plot(numbers_test_Y_string.split(),predict[i], marker= 'o', linestyle = '', c = 'b'  ,alpha = 0.5)
        # # plt.ion()
    plt.xlabel('data')
    plt.ylabel('prediction')
    plt.yticks(range(0,10))
    plt.legend(handles = [], loc=1)
plt.show()







# add crosvalidation