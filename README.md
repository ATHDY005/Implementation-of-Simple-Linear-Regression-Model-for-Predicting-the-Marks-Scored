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
## Head and Tail Values 

![image](https://user-images.githubusercontent.com/84709944/229363842-3dc23810-93d9-4cbe-96f4-6087dc344921.png)

## Array value of X

![image](https://user-images.githubusercontent.com/84709944/229363860-ec9fc0b3-1805-44cc-8f47-ba1cc62aaec4.png)

## Array value of Y

![image](https://user-images.githubusercontent.com/84709944/229363877-86ebe1be-fc69-4efc-9fab-7b2c5876b313.png)

## Value of Y Prediction

![image](https://user-images.githubusercontent.com/84709944/229363892-fd224268-41f7-49e6-b525-3e2a3f8469ab.png)

## Array value of Y_test

![image](https://user-images.githubusercontent.com/84709944/229364562-9319502e-aaf3-4f9e-b622-c3d90920bf8c.png)


## TRAINING SET

![image](https://user-images.githubusercontent.com/84709944/229363474-b860eda1-12c4-4bb0-bda4-6ce22d9a3df5.png)

## TEST SET

![image](https://user-images.githubusercontent.com/84709944/229363497-b5fe2e9f-7cdd-43e8-b2ea-dc935b06ca7a.png)

## VALUES OF MSE,MAE,RMSE

![WhatsApp Image 2023-04-02 at 9 41 22 PM](https://user-images.githubusercontent.com/84709944/229422595-7f37b94e-25e5-4ccc-bd96-b8ffd97920f8.jpeg)


## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
