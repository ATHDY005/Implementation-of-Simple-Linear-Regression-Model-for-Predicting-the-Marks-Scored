# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Read the given dataset
2. Assign values for x and y and plot them
3. Split the dataset into train and test data
4. Import linear regression and train the
data
5. find Y predict
6. Plot train and test data
7. Calculate mse,mae and rmse


## Program:

/*

Program to implement the simple linear regression model for predicting the marks scored.

Developed by: Harini Shamlin

RegisterNumber:  212220040040


import numpy as np

import pandas as pd

dataset=pd.read_csv('/content/student_scores.csv')

dataset.head()

dataset.tail()

x=dataset.iloc[:,:-1].values

y=dataset.iloc[:,1].values

print(x)

print(y)

import matplotlib.pyplot as np

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=2/3,random_state=0)

from sklearn.linear_model import LinearRegression

reg=LinearRegression()

reg.fit(x_train,y_train)

y_predict=reg.predict(x_test)

import matplotlib.pyplot as plt

from sklearn.metrics import mean_absolute_error, mean_squared_error

plt.scatter(x_train,y_train,color='grey')

plt.plot(x_train,reg.predict(x_train),color='magenta')

plt.title('Training set(H vs S)')

plt.xlabel('Hours')

plt.ylabel('Scores')

plt.show()

plt.scatter(x_test,y_test,color='magenta')

plt.plot(x_test,reg.predict(x_test),color='grey')

plt.title('Test set(H vs S)')

plt.xlabel('Hours')plt.ylabel('Scores')

plt.show()

print(y_predict)

plt.show()

print(y_predict)

mse=mean_squared_error(y_test,y_predict)

print('MSE = ',mse)

mae=mean_absolute_error(y_test,y_predict)

print('MAE = ',mae)

from sklearn.metrics import mean_squared_error

from math import sqrt

sqrt(mean_squared_error(y_test,y_predict)) 

*/

## Output:
![image](https://user-images.githubusercontent.com/84709944/229363474-b860eda1-12c4-4bb0-bda4-6ce22d9a3df5.png)
![image](https://user-images.githubusercontent.com/84709944/229363497-b5fe2e9f-7cdd-43e8-b2ea-dc935b06ca7a.png)



## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
