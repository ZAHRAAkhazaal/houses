# -*- coding: utf-8 -*-
#Import Libraries
#import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
 
from sklearn.linear_model import LinearRegression

url = "https://raw.githubusercontent.com/DrSaadLa/MLLabs/main/data/housing.csv"

df=pd.read_csv(url)
#df.float()


print(df.head())
print(df.describe())
print(df.info())
print(df.tail())



plt.show(sns.pairplot(df))

#distplot stands for distribution plot
plt.show(sns.distplot(df['Price']))

#df.corr() shows corelation between all the columns
print(df.corr())

plt.show(sns.heatmap(df.corr()))

plt.show(sns.heatmap(df.corr(),annot=True))

print(df.columns)


X=df[['Avg. Area Income', 'Avg. Area House Age', 'Avg. Area Number of Rooms',
       'Avg. Area Number of Bedrooms', 'Area Population']]

print(X)

#y is our taget variable.As we want to predict here house prices,so it will contain price column values
y=df['Price']
print(y)











#--------------------------------------
#sns.set(rc={'figure.figsize':(10,7)})
#sns.heatmap(df.corr(), cmap='RdYlBu', square=True)
#-----------------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 44)


X_train
X_test
y_train
y_test 




linearregressionmodel=LinearRegression()
linearregressionmodel.fit(X_train, y_train)
#3--------------------------------------------------


y_pred=linearregressionmodel.predict(X_test)


#---------------------------------------

print('scoretrain', linearregressionmodel.score(X_train, y_train))
print('scoretest',linearregressionmodel.score(X_test,y_test))




from sklearn.metrics import mean_absolute_error
mean_absolute_error(y_test, y_pred)

from sklearn.metrics import mean_squared_error
mean_squared_error(y_test, y_pred)
