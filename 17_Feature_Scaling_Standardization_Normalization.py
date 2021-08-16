# (17)*******************Feature Scaling in Standardization And Normalization in Data Preprocessing and Feature Engineering***********************

# ----Discuss About----
# ----What is Standardization?
# ----What is Normalization?
# ----Standardization vs Normalization
# ----Practical


# *********What is Standardization
# ----Definition-----Standardization rescale the feature such as mean()=0 and standard deviation()=1
# ----Definition 2----Data standardization is the process of rescaling one or more attributes so that they have a mean value of 0 and a standard deviation of 1. Standardization assumes that your data has a Gaussian (bell curve) distribution.s
# ----Formula----z=(x-mean)/std
# ----If data follow normal distribution(gaussian distribution).
# ----If the origi nal distribution is normal, then the standardised distribution will be normal
# ----If the original distribution is skewed, then the standardised distribution of the variable will also be skewed.


# *********What is Normalization
# ----Definition---Normalizarion rescale the feature in fixes range between 0 to 1.
# ----Definition---Normalization is a technique often applied as part of data preparation for machine learning. The goal of normalization is to change the values of numeric columns in the dataset to use a common scale, without distorting differences in the ranges of values or losing information.
# ----Normalization also called as Min-Max Scaling.
# ----Formula---Xnorm=(X-Xmin)/(Xmax -Xmin)
# ----If data doesn't follow normal distribution (Gaussian Distribution(bell curve))


# *************Standardization vs Normalization?
# ----There is no any thumb rule to use Standardization or Normalization for Special ML algo
# ----But mostly Standardization use for clustering analyses,Principal Component Analysis(PCA)
# ----Normalization prefer for image processing because pixel intensity between 0 to 255, neural network algorithm require data in scale 0-1,K-Nearest Neighbors.


# ****************Types of Feature Scaling
# ----(1)-Min Max Sclaer-----Maximum use---depend on type of algorithm
# ----(2)-Standard Scaler----Most and maximum use----depend on algorithm
# ----(3)-Max Abs Scaler
# ----(4)-Robust Scaler
# ----(5)-Quantile Transformer Scaler
# ----(6)-Power Transformere Scaler
# ----(7)-Unit Vector Scaler


# ***********Practical

# -----importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

# -----importing library from github
df=sns.load_dataset('titanic')
# print(df.head())

# ----we use just few columns 
df2=df[['survived','pclass','age','parch']]
# print(df2.head())

# ----for cleaning the dataframe
df3=df2.fillna(df2.mean())
X=df3.drop('survived',axis=1)
y=df3['survived']
# print('shape of X is:',X.shape)
# print('shape of y is:',y.shape)

# -----Now we split the train and test data because machine learning algo training have 2 variable 1st is train data which is achual and read data and 2nd is test data which we use as a test
# -----So we split the real data into train and test data by using of scikit learn library
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=51)
# print('Shape of X_train is: ',X_train.shape)
# print('shape of y_train is: ',y_train.shape)
# print('shape of X_test is: ',X_test.shape)
# print('shape of y_test is: ',y_test.shape)


# ------Now we use standard Scaler function in sklearn
sc=StandardScaler()
sc_fit=sc.fit(X_train)
# print(sc_fit)
# print(sc.mean_)
# print(sc.scale_)
# print(X_train.describe())


# ----Now we transform the data which we standardized it
X_train=sc.transform(X_train)
X_test=sc.transform(X_test)
# print(X_train)
# print(X_test)

# -----Now we convert the array to dataframe because sklearn is returning the values as a array
X_train_sc_con=pd.DataFrame(X_train,columns=['pclass','age','parch'])
X_test_sc_con=pd.DataFrame(X_test,columns=['pclass','age','parch'])
# print(X_train_sc_con)
# print(X_test_sc_con )

# ----Now printing the describe of this dataframe
# print(X_train_sc_con.describe())
# ----if we want to describe in the round function then we use round()function
# print(X_train_sc_con.describe().round(2))


# -----Now we apply MinMaxScaler function in sklearn library
mmc=MinMaxScaler()
# -----We fit the datasets and transform data 
X_train_mmc=mmc.fit(X_train).transform(X_train)
X_test_mmc=mmc.fit(X_test).transform(X_test)
# print(X_train_mmc)
# print(X_test_mmc)


# --------Now we transform the data
# -----this is not working
# X_train_mmc_tran=mmc.transform(X_train_mmc)
# X_test_mmc_tran=mmc.transform(X_test_mmc)
# print(X_train_mmc_tran)
# print(X_test_mmc_tran)

# ----Now we converd the array data to dataframe
X_train_mmc_df=pd.DataFrame(X_train_mmc,columns=['pclass','age','parch'])
X_test_mmc_df=pd.DataFrame(X_test_mmc,columns=['pclass','age','parch'])
# print(X_train_mmc_df)
# print(X_test_mmc_df)

# -----checking the describe of the changed dataframe
print(X_train_mmc_df.describe())
print(X_train_mmc_df.describe())

 