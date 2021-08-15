# (12)*************************Fill values using sklearn library technique in the DataFrame Handling Missing Values in Data Preprocessing and Feature Engineering***********************************

# -----we use SimpleImputer function in sklearn to fill the nan values
# -----sklearn is the very powerfull library in python 
# -----in this tutprial we discuss about how to fill nan values using sklearn(numarical and categorical)

# -----we fill values manually as well as Global constant as well as mean median and mode
# ------if data found numarical than we use mean or median 
# ------if data found categorical than we use mode function

# -----importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer

# -----importing datasets
train=pd.read_csv(r'datasets/Housing Price/train.csv')
test=pd.read_csv(r'datasets/Housing Price/test.csv')
# print(train.head())
# print(test.head())

# ----for showing the shape of the datasets we use shape function
# print('shape of train df =',train.shape)
# print('shape of test df =',test.shape)

# ----droping the target variable of train data because target variable is not present in the test df
# ----and we assiging the target variable of the other variable 
# ----train data take input and different output
# ----test data has taked only just input not output

X_train=train.drop(columns='SalePrice')
y_train=train['SalePrice']

# -----Now showing the shape of the train variables
# print('shape of X_train df =',X_train.shape)
# print('shape of y_train df =',y_train.shape)

# ******************Now Filling the Numarical values of the train and test df using sklearn library
# -----selecting numarical columns
num_vars=X_train.select_dtypes(include=['int64','float64']).columns
# print(num_vars)

# -----Now check that the null values of numarical variables
# print(X_train[num_vars].isnull().sum())

# ----Now we use sklearn library to filling the nan values
# ----if we change the values as we wish than we input in the place of missing_vlaues='?'
# ----we use strategy parameter which we input mean,median,mode,constant(default=mean)
# ----first of all we imputing the values
# -----Now sklearn is making the blue print and take it
# ----we use this  function when we need
imputer_mean=SimpleImputer(strategy='mean')
# -----if we want to put values as we wish than we use this method
# imputer_mean2=SimpleImputer(strategy='constant',fill_values='Missing')
# -----fit is a function which is input the dataframe and also numaric data 
sk_fit=imputer_mean.fit(X_train[num_vars])
# print(sk_fit)
# -----statistics parameter isreturning the values of place all the nan vlaues
sk_stat=imputer_mean.statistics_
# print(sk_stat)

# ----Now we tranform the data in the real dataFrame
# ----disadvantage-----this is imputing the values but in the form of arrays
# sk_trans=imputer_mean.transform(X_train[num_vars])
# print(sk_trans)

# ----Now converting the array to dataFrame as simple as that we taking variable of copy
X_train[num_vars]=imputer_mean.transform(X_train[num_vars])
# ----we also imputing values in the test dataFrame
test[num_vars]=imputer_mean.transform(test[num_vars])

# /-----Now checking that the X_train and test df are exists the nan values or not
print(X_train[num_vars].isnull().sum())
print(test[num_vars].isnull().sum())






# ----