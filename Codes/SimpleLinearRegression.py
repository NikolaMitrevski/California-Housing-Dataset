#Import Necessary Libraries:
import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder,StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor 
from sklearn.svm import SVR

import statsmodels.formula.api as smf

from sklearn.metrics import mean_squared_error,r2_score
from math import sqrt

import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline

import warnings
warnings.filterwarnings('ignore')

from matplotlib.axes._axes import _log as matplotlib_axes_logger
matplotlib_axes_logger.setLevel('ERROR')

from sklearn.model_selection import train_test_split

####################################################

df_house = pd.read_csv('housing.csv')
df_house.head()

####################################################

data_refine = df_house.drop('ocean_proximity', axis = 1)
data_refine.head()

####################################################

data_refine.isnull().sum()

####################################################

#removing NA/NaN values
data_refine = data_refine.dropna(axis = 0)
data_refine.isnull().sum()

####################################################

X = data_refine.drop('median_house_value', axis = 1)
Y = data_refine['median_house_value']

X = X[['total_bedrooms']]

print(data_refine.shape)
print(X.shape)
print(Y.shape)

####################################################

plt.scatter(X['total_bedrooms'], Y)
plt.xlabel('total_bedrooms')
plt.ylabel('House Price')

####################################################

X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.25)
print(X_train.shape)
print(Y_train.shape)
print(X_test.shape)
print(Y_test.shape)

####################################################

LR = LinearRegression()
LR.fit(X_train, Y_train)

####################################################

predict = LR.predict(X_test)
print('Predicted Value :',predict[3])
print('Actual Value :',Y_test.values[3])

####################################################

print(sqrt(mean_squared_error(Y_test, predict)))
print((r2_score(Y_test, predict)))

####################################################

gr = pd.DataFrame({'Predicted':predict,'Actual':Y_test})
gr = gr.reset_index()
gr = gr.drop(['index'],axis=1)
plt.plot(gr[:1000])
plt.legend(['Actual','Predicted'])
#gr.plot.bar();