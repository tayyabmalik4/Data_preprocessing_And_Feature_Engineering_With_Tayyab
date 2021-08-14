# (10)************************Fill the Values by class technique of missing values in Data Preprocessing and Feature Engineering**********************
# ----this is the most correct method to filling missing values---
# ----by the help of class method we categories the class and than we take mean or median and put it

# -----About-----
# ---(1)-Fill missing value manually
# ---(2)-Global constant
# ---(3)-Measure of central tendency for each class


# ----importing libraries
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ----importing csv file from google drive
file_link="https://drive.google.com/file/d/1dGgHEwbOnBDgBpP98QmiAZO9rKcDm2tA/view?usp=sharing"
linked="https://drive.google.com/uc?export=download&id="+file_link.split('/')[-2]
df=pd.read_csv(linked)
# print(df.head())

# -----for chacking the shape 
# print(df.shape)

# ----for displaying all the rows and columns we use this function
pd.set_option('display.max_columns',None)
pd.set_option('display.max_rows',None)
# print(df.head())

# ----for checking the null values for all collumns
# print(df.isnull().sum())

# -----for showing the percentage of nan values for all columns
per=df.isnull().sum()/df.shape[0]*100
# print(per)

# -----checking these columns who the empty values are present in 20% or greater than 20%
df1_show_clm=per[per>20].keys()
# print(df1_show_clm)

# ----droping these columns who the empty values are present greater than 20%
df2_drop_clm=df.drop(columns=df1_show_clm)
# print(df2_drop_clm.shape)


# ****************Select numarical variables of the dataset
# -----for giving up the numaric datatypes
df3_num=df2_drop_clm.select_dtypes(include=['int64','float64'])
# print(df3_num.info())

# print(df3_num.isnull().sum())

# ----for giving up these columns of numaric data which presents the empty values greater than 0
df4_num_emp=[var for var in df3_num.columns if df3_num[var].isnull().sum()>0]
# print(df4_num_emp)

# ----for showing the data of these dataframe which present the empty values
# print(df3_num[df4_num_emp][df3_num[df4_num_emp].isnull().any(axis=1)])


# -----if we want to fill the values by helping the classes we always know the domain knowledge
# -----but now we just comaring the column names and fill the values by class

# ----now we compare the LotFrontage columns to LotConfig columns 
# ----and we printout that the classes of LotConfig
# ----unique function is help to printout the classes who are unique
# print(df['LotConfig'].unique())


# **********************Now we build a logic to creat filling the vlaues by classes 
# ----first of all we creat a variabel who the location is define
# ----in this variable we use loc function to define the location and in this function we take all rows but one variabel who is comparinng it
locations=df[df.loc[:,'LotConfig']=='Inside']["LotFrontage"]
# print(locations)
# ----now we replacing the thhe nan values to mean or median of the other values by class
replac=locations.replace(np.nan,locations.mean())
# print(replac)

# -----Now we build a logic for all the classes to fill the mean values using for loop
# -----we copied the real data frame
df_copy=df.copy()
# for var_class in df['LotConfig'].unique():
#     df_copy.update(df[df.loc[:,'LotConfig']==var_class]['LotFrontage'].replace(np.nan,df[df.loc[:,'LotConfig']==var_class]['LotFrontage'].mean()))
# print(df_copy.isnull().sum())


# -----Now we fill the multiple columns to fill the empty values for using this logic using double for loop
# ----this is these columns who fill the values
# num_vars_miss=['LotFrontage', 'MasVnrArea', 'GarageYrBlt']
# ---this is these columns which we help that to fill vlaues to creating the classes----And for this variables is use when we domain knowledge----but now we just let that 
cat_vars=['LotConfig','MasVnrType','GarageType']
# ----we use for loop as well as zip function to combine these columns step by step
# for cat_var,num_var_miss in zip(cat_vars,num_vars_miss):
#     for var_class in df[cat_var].unique():
#         df_copy.update(df[df.loc[:,cat_var]==var_class][num_var_miss].replace(np.nan,df[df.loc[:,cat_var]==var_class][num_var_miss].mean()))

# ----Now we check that the values are filled or not
# ----this shows that the values are empty yet because the vlaues of cat_vars is also empty so be carefull to filling the values
# print(df_copy[num_vars_miss].isnull().sum())
# -----for checking the empty values of MasVnrType
# print(df_copy[df_copy[['MasVnrType']].isnull().any(axis=1)])


# -----Now we printout by the other cat_vars
num_vars_miss=['LotFrontage', 'MasVnrArea', 'GarageYrBlt']
cat_vars=['LotConfig','Exterior2nd','KitchenQual'] 
for cat_var,num_var_miss in zip(cat_vars,num_vars_miss):
    for var_class in df[cat_var].unique():
        df_copy.update(df[df.loc[:,cat_var]==var_class][num_var_miss].replace(np.nan,df[df.loc[:,cat_var]==var_class][num_var_miss].mean()))
# print(df_copy[num_vars_miss].isnull().sum())

# -----For checking the difference of Real Dataset and modified dataset we use subplot and also for loop
plt.figure(figsize=(13,13))
sns.set()
for i,var in enumerate(num_vars_miss):
    plt.subplot(2,2,i+1)
    sns.distplot(df[var],bins=20,kde_kws={'linewidth':8,'color':'r'},label='Original')
    sns.distplot(df_copy[var],bins=20,kde_kws={'linewidth':5,'color':'g'},label='Mean')
    plt.legend()
plt.show()
