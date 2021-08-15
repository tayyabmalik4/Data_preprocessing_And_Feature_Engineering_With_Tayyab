# (15)****************Label Encoding and also learn Ordinal Encoding using sklearnn library in Data Preprocessing and Feature Engineering******************

# ----Definition Ordinal Variables----Ordinal Variables are categorical variables in which the categories can be meaningfully ordered
# ----When the ordinal variable convert into the number then it has mathematical values
# ----For Example----Grade,month's name


# -----Label Encoding Definition----Label Encoding apply on ordinal and nominal categorical variables----this is the impact of machine learing so we do not use this function we use Ordinal Encoding

# -----when we convert the nominal data converts numaric data than this is picking as a Word wise----for example--- A refers to 1 and B refers to 2 etc

# /------Ordinal Encoding Definition----Ordinal Encoding apply on ordinal categorical variables.


# -----In this session we discuss about who to encode of ordinal variables using LabelEncode function in sklearn library
# -----for example-----we convert Grade(A,B,C,D) to Grade(1,2,3,4)
# -----as you know the machine is not understand the Categorical variables and we converd the categorical variables to Numaric variables

# ----importing libraries
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

# ----importing dataset
df=pd.read_csv(r"datasets/Housing Price/train.csv")
pd.set_option('display.max_columns',None)
pd.set_option('display.max_rows',None)
# print(df.head())

# -----we work just 2 columns KitchenQual and BldgType 
df2=df[['KitchenQual','BldgType']]

# -----starting LabelEncoding in Categoricla nominal variable BldgType
le=LabelEncoder()
le1=le.fit_transform(df2["BldgType"])
# print(le1)

# ----Now we converd dataFrame
df2['BldgType_L_enc']=le.fit_transform(df2['BldgType'])
# print(df2)

# ----now we count the total values of dataframe BldgType column
# print(df['BldgType'].value_counts())


# -----------Now we learn about Ordianry dataframe
# ----counts the values of dataframe KitenQual
print(df['KitchenQual'].value_counts())

# ----these types are the KitchenQual of dataset
# Ex     ------------Excellent
# Gd     ------------Good
# TA     ------------Typical/Average
# Fa     ------------Fair

# -----Now we put values as we wish in the place of order_label
order_label={'Ex':4,'Gd':3,'TA':2,'Fa':1}
df2['KitchenQual_org_enc']=df2['KitchenQual'].map(order_label)
print(df2)