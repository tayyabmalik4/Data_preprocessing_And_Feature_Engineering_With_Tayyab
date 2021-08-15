# (14)**********************One Hot Encoding Dummy variables in Categorical variables in data preprocessing and Feature Engineering**************************

# -----One Hot Encoding Dummy variables mean that if Categorical variables are Nominal(not-ordinary) than we converd the 0 or 1 form to understanding the Machine Learning Algorithms

# -----For example ----columns name is sex and the sex is female or male than we convert female to 0 and male to 1 and
# ----And this function is exists in pandas library pd.get_dummies(dataset)
# ----this is the very very helpfull for training the algorithms

# -----importing librairs
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

# -----importing dataset
df=pd.read_csv('datasets/hotal dataset/tips.csv')
# print(df.head())
df1=df.drop(columns=['Payer Name','CC Number','Payment ID'],inplace=True)
df2=pd.get_dummies(df)
# print(df2.head())
# print(df2.keys())

# -----if we want to creat just one Hot Encoding than we use pd.get_dummies(dataframe,drop_first=True)
# -----in the drop_first parameter we increase the ability of machine learning
df3=pd.get_dummies(df,drop_first=True)
# print(df3)

# ************************Scikit-learn***********************
# -----we convert the dummy values by using the sklearn function OneHotEncoder
oh_enc=OneHotEncoder(sparse=False)
# ------now we tranform the data using fit_transform function
oh_enc_fit=oh_enc.fit_transform(df[['sex','smoker','day','time']])
# -----this is output as a array so this is one of the disadvantage of sklearn
# print(oh_enc_fit)

# -----Now we convert the array to dataFrame
converter=pd.DataFrame(oh_enc_fit,columns=['sex_Female','sex_Male', 'smoker_No', 'smoker_Yes', 'day_Fri', 'day_Sat', 'day_Sun','day_Thur', 'time_Dinner', 'time_Lunch'])
print(converter)
