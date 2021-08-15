# (13)-*************************Fill Missing Values using sklearn library Hanling Missing Values in Data Preprocessing and Feature Engineering**********************************

# -----when we want to fill the values as we wish i mean when some values are fill in the mean function and some fill by median function and some fill by mode(most_frequent) and also some fill by constant than we use sklearn library and also use of pipeline,SimpleImputation and ColumnTransformer functions

# ----Missing values Imputation(Numarical and Categorical)
# ----Strategy for Different Variables(SimpleImputater,Pipeline,ColumnTransformer)

# ----importing librariries
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


# -----import dataset files
train=pd.read_csv(r"datasets/Housing Price/train.csv")
test=pd.read_csv(r"datasets/Housing Price/test.csv")
# print('shape of train df =',train.shape)
# print('shape of test df = ',test.shape)

# ----Now we changing the datasets
X_train=train.drop(columns='SalePrice',axis=1)
y_train=train["SalePrice"]
X_test=test.copy()
# print("shape of X_train =",X_train.shape)
# print("shape of y_train = ",y_train.shape)
# print("shape of X_test = ",X_test.shape)


# -----Now we check the null values
isnull_values=X_train.isnull().sum()
# print(isnull_values)


# ----Now we finding the numarical variables 
num_vars=X_train.select_dtypes(include=['int64','float64']).columns
# ----Now we printout these numarical columns who have the empty values presents 0% or greater than 0%
num_vars_miss=[var for var in num_vars if isnull_values[var]>0]
# print(num_vars_miss)
num_vars_emp=['LotFrontage', 'MasVnrArea', 'GarageYrBlt']

# ----Now we finding the categorical variables
cat_vars=X_train.select_dtypes(include='object').columns
# ----for printout these categorical columns who have the empty values present 0% or greater than 0%
cat_vars_miss=[var for var in cat_vars if isnull_values[var]>0]
# print(cat_vars_miss)
cat_vars_emp=['Alley', 'MasVnrType', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'Electrical', 'FireplaceQu', 'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond', 'PoolQC', 'Fence', 'MiscFeature']


# -----Now we creat a pipeline for filling the empty values
# -----creating mean variable which we take these columns who want to fill empty values in the mean
num_var_mean=['LotFrontage']
num_var_median=['MasVnrArea','GarageYrBlt']
num_var_mode=['Alley', 'MasVnrType', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'Electrical', 'FireplaceQu']
num_var_constant=['GarageType', 'GarageFinish', 'GarageQual', 'GarageCond', 'PoolQC', 'Fence', 'MiscFeature']

# ----Now creating Pipeline class
num_var_mean_imputer=Pipeline(steps=[('imputer',SimpleImputer(strategy='mean'))])
num_var_median_imputer=Pipeline(steps=[("imputer",SimpleImputer(strategy="median"))])
num_var_mode_imputer=Pipeline(steps=[("imputer",SimpleImputer(strategy="most_frequent"))])
num_var_constant_imputer=Pipeline(steps=[("imputer",SimpleImputer(strategy="constant",fill_value="Missing"))])

# -----Now cincated the imputer and variable
preprocesser=ColumnTransformer(transformers=[('mean_imputer',num_var_mean_imputer,num_var_mean),("median_imputer",num_var_median_imputer,num_var_median),("mode_imputer",num_var_mode_imputer,num_var_mode),("constant_imputer",num_var_constant_imputer,num_var_constant)])

# -----now fit the proprocesser
df_fit=preprocesser.fit(X_train)
# print(df_fit)

# -----Now we check values which we take it so we use this method
values_checked=preprocesser.named_transformers_["mean_imputer"].named_steps["imputer"].statistics_ 
# print(values_checked)
# -----Now check the mode of the columns which we finded it
check_mode=preprocesser.named_transformers_["mode_imputer"].named_steps['imputer'].statistics_
# print(check_mode)

# -----check that the mean value of real dataframe LotFrontage
# print(train['LotFrontage'].mean())

# -----Now we transform the the making dataframe to real dataframe
X_train_clean=preprocesser.transform(X_train)
X_test_clean=preprocesser.transform(X_test)
# print(X_train_clean)

# ------for checking the changes of preprocessing dataframe
# ------the sklearn is droping these indexes which is exist empty values if we don't drop the indexes we use passthrough function
# print(preprocesser.transformers_)

# ------ NOw we converd this array to DataFrame
X_train_clean_miss_var=pd.DataFrame(X_train_clean,columns=num_var_mean+num_var_median+num_var_mode+num_var_constant)
# print(X_train_clean_miss_var.head())
print(X_train_clean_miss_var.shape)

# ----checking that the data achully added or not then we use this method
print(train["Alley"].value_counts())
print(X_train_clean_miss_var["Alley"].value_counts())
print(X_train_clean_miss_var["MiscFeature"].value_counts())
