# *******************************Fill Values by helping Mode function technique part 05 of the Cleaning Missing Values in Data Preprocessing and Feature Engineering***************************

# ------Mode Definition----Mode means the maximum value of the column
# ------When values are present in randomly than we use these techniques
# ------And also when our data is missing near about 5% than we use this method
# ------We use Mode just when the column is categorical

# -----importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ----importing dataset
df=pd.read_csv('datasets/Housing Price/train.csv')
# print(df.head())

# for displaying all the columns and rows we use set_option function
pd.set_option('display.max_columns',None)
pd.set_option('display.max_rows',None)

# -----when we need just categorical data than we use this function
df2_cat=df.select_dtypes(include='object')
# print(df2_cat.shape)
# print(df2_cat.info())

# -----for checking the percentage of nan values
per=df2_cat.isnull().sum()/df2_cat.shape[0]*100
# print(per)
# -----2nd way to check percentage
per2=df2_cat.isnull().mean()*100
# print(per2 )

# -----Now we show out these columns who is present nan values 20% or greater than 20%
df3_drop_20per=per[per>20].keys()
# print(df3_drop_20per)

# ----Now we drop these columns who the nan values presents 20% or greater than 20%
df4_drop_clm=df2_cat.drop(columns=df3_drop_20per,inplace=True)
# print(df4_drop_clm.shape)

# ----Now printout these columns who the nan values presents greater than 0%
# ----first of all we given percentage of df4_drop_clm
df4_drop_clm_per=df2_cat.isnull().sum()/df2_cat.shape[0]*100
# print(df4_drop_clm_per)
# -----2nd way to printout percentage
# df4_drop_clm_per2=df4_drop_clm.isnull().mean()*100
# print(df4_drop_clm_per2)

# ----Now we given whose columns who the nan values are present greater than 0%
df5_cat_0per=df4_drop_clm_per[df4_drop_clm_per>0].keys()
# print(df5_cat_0per)

# *****************Now if we want to fill nan values as a Global contant than we use this method
fill_nan_df5=df2_cat['MasVnrType'].fillna("Missing")
# print(df5_cat_0per)
# print(fill_nan_df5)

# -----But now we fill values of categorical variables as a mode we use this function
# cat_fill_mode=df2_cat['MasVnrType'].mode()

# -----for counting the values class by class we use this method
checking_count=df2_cat['MasVnrType'].value_counts()
# print(checking_count)

# -----Now fill the values as a mode
cat_fill_nan_mode=df2_cat['MasVnrType'].fillna(df2_cat['MasVnrType'].mode()[0])


# -----Now we write code to fill nan values for all the columns using for loop
for var in df5_cat_0per:
    df2_cat[var].fillna(df2_cat[var].mode()[0],inplace=True)
    # ----for shhowing that who values are changed
    # print(var +"="+ df2_cat[var].mode()[0])

# print(df2_cat.isnull().sum())


# -----Now we check the distribution of datasets(Real and after cleaning)
plt.figure(figsize=(13,13))
sns.set()
for i,var in enumerate(df5_cat_0per):
    plt.subplot(4,3,i+1)
    plt.hist(df2_cat[var],label='Mode')
    plt.hist(df[var].dropna(),label='Real')
    plt.legend()
# plt.show()

# -----Now we update the real dataFrame
df.update(df2_cat)
df.drop(columns=df3_drop_20per,inplace=True)

# -----Now checking that the dataframe is achully updated or not
updated=df.select_dtypes(include='object').isnull().sum()
print(updated)

