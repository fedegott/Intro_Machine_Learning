from sklearn import datasets
from sklearn import linear_model
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pandas.plotting import scatter_matrix


cali = datasets.fetch_california_housing()
print(type(cali))

#plt.imsho

cal = pd.DataFrame( data = cali.data, columns = cali.feature_names)
print(cal.describe(), cal.head(10), cal.info()) # head(50) shows the top 50, info shows #data points and data type for each feature
# cal.hist(bins = 50, figsize=(10,10))
# plt.show()

# cal.plot( kind = "scatter", x = cal['Longitude'], y = cal['Latitude'])
# plt.show()

# plt.scatter(x = cal['Longitude'], y = cal['Latitude'], c = np.log10(cal['Population']), cmap = 'viridis', linewidth = 0.5, alpha = 0.75, s = cal['Population']/100, label = "population")
# plt.axis(aspect = 'equal')
# plt.xlabel('longitude')
# plt.ylabel('latitude')
# plt.colorbar(label = 'log$_{10}$(Population')
# plt.clim(2,5)
# plt.show()


# plt.scatter(x = cal['Longitude'], y = cal['Latitude'], c = cal['MedInc'], cmap = plt.get_cmap('jet'), linewidth = 0.5, alpha = 0.4, s = cal['Population']/100, label = "population")
# plt.axis(aspect = 'equal')
# plt.xlabel('longitude')
# plt.ylabel('latitude')
# plt.colorbar(label = 'Median Income')
# plt.clim(2,5)
# plt.show()

# attributes = cal.columns
# scatter_matrix(cal[attributes], figsize=(20,15))
# plt.show()

corr_matrix = cal.corr(method = 'pearson')
print(corr_matrix)
print(corr_matrix['MedInc']>0.5) #fancy indexing because use boolean to select data

np.random.seed(0)
indices = np.random.permutation(len(cal['MedInc']))
cal_train = cal['MedInc'][indices[:-1000]]
# print(cal_train)


lnn = linear_model.LinearRegression()
# lnn.fit(cal_train['HouseAge'], cal_train['MedInc'])
