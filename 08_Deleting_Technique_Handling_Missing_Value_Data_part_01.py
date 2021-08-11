# (08)**********************Handling Missing Values for using Deleting technique in Data Preprocessing and Feature Engineering*******************************

# -----Ignore missing values row / Delete row
# -----We delete these colums who 20% or greater than 20% is empty or nan values
# -----this is the general Rule
# -----But also it base the Data or different satuations

# ////importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# //////load csv file
df=pd.read_csv('datasets/Housing Price/train.csv')
# print(df)

# ----For checking the shape of the dataframe we use shape function----how many colums and rows are present
# print(df.shape)

# ----for printing the first few rows we use head function
# print(df.head(6))

# ----for printing the last few rows of data frame we use tail function
# print(df.tail(6))

# ----for showing all the columns and rows we use set_option function
pd.set_option('display.max_columns',None)
pd.set_option('display.max_rows',None)
# print(df.tail())

# -----for giving the information of data set we use info() function
# print(df.info())

# -----for checking that who many columns are presents the empty(nan) values
# print(df.isnull().sum())

# -----for printing the graph to knowing the values we use heatmap function in seaborn
# -----for increases the figure size we use plt.figure function
# -----White color shows the empty values and black shows the real values
# plt.figure(figsize=(13,13))
# sns.heatmap(df.isnull())
# plt.show()

# -----For showing the percentage we use the formula
# -----Formula----first of all we take the null values of the dataFrame than we take the shape of the dataFrame(0 represents the axis-rows) and than multiply by 100-----
percentage=(df.isnull().sum()/df.shape[0])*100
# print(percentage)

# ----Now we drop these columns who the empty(NAN) values are present 20% and greater than 20%
# ----But one column is showing the 17% empty values and now we drop 17% and grater than 17% empty values
drop_columns=percentage[percentage>17].keys()
# print(drop_columns)
# -----finally droping these columns
df2_drop_clm=df.drop(columns=drop_columns)
# print(df2_drop_clm)



# ----now checking the shape after droping the columns
# print(df2_drop_clm.shape)

# ----plotting the heatmap graph using seaborn after droping the columns
# sns.heatmap(df2_drop_clm.isnull())
# plt.show()

# ----Now we drop those rows who exist the empty values
# ----in the dropna function it is default rows values 
df3_drop_rows=df2_drop_clm.dropna() 
# print(df3_drop_rows)
# print(df3_drop_rows.shape)

# ----Now plotting heatmap graph using seaborn after droping the rows and columns
# plt.figure(figsize=(13,13))
# sns.heatmap(df3_drop_rows.isnull())
# plt.show()


# -----Now to showing the empty values in numarical form 
# -----this is the clear for all null values in the dataFrame
# print(df3_drop_rows.isnull().sum())


# ----now we know that we are correct or not thats why we use distplot in seaborn to showing the graph
# ----and dist function is take just numaric values now first of all we given the numaric values  to take some functions
numaric=df3_drop_rows.select_dtypes(include=['int64','float64']).columns
# print(numaric)
# ----now showing distplot graph using seaborn
# sns.distplot(df['MSSubClass'])
# ----now we show the distplot after deleting nan values
# sns.distplot(df3_drop_rows['MSSubClass'])

# ----now we overlapping the graphs to showing the accuracy
# sns.distplot(df['MSSubClass'])
# sns.distplot(df3_drop_rows['MSSubClass'])
# plt.show()

# ----For plotting all of the numaric values graph
# ----first of all we assign all the numaric values in a list
num_val=['MSSubClass', 'LotArea', 'OverallQual', 'OverallCond',
       'YearBuilt', 'YearRemodAdd', 'MasVnrArea', 'BsmtFinSF1', 'BsmtFinSF2',
       'BsmtUnfSF', 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 'LowQualFinSF',
       'GrLivArea', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath',
       'BedroomAbvGr', 'KitchenAbvGr', 'TotRmsAbvGrd', 'Fireplaces',
       'GarageYrBlt', 'GarageCars', 'GarageArea', 'WoodDeckSF', 'OpenPorchSF',
       'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea', 'MiscVal',
       'MoSold', 'YrSold', 'SalePrice']
# ----for increasing the figure size
# plt.figure(figsize=(13,13))
# ----Now we use for loop to itrating all the rows 
# ----enumerate function is taking the indexes automatically from 0 to onward----for easy to use python we use enumerate function
# for item,items in enumerate(num_val):
    # ----Now we plotting all the graphs we use subplot in matplotlib
    # ----sublopt is very helpful for creating more than two graphs at a time
    # ----subplot first of all taking how many rows than how many columns and than indexes
    # plt.subplot(9,4,item+1)
    # ----now we creating the graphs and comparison of the real and drop_out dataFrames
    # sns.distplot(df[items],bins=20)
    # sns.distplot(df3_drop_rows[items],bins=20)
# plt.show()

# ----Now we showing the categorical data and Represent it
# ----first of all showing all the categorical values
cat_df=df3_drop_rows.select_dtypes('object').columns
# print(cat_df)

categorical=['MSZoning', 'Street', 'LotShape', 'LandContour', 'Utilities',
       'LotConfig', 'LandSlope', 'Neighborhood', 'Condition1', 'Condition2',
       'BldgType', 'HouseStyle', 'RoofStyle', 'RoofMatl', 'Exterior1st',
       'Exterior2nd', 'MasVnrType', 'ExterQual', 'ExterCond', 'Foundation',
       'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2',
       'Heating', 'HeatingQC', 'CentralAir', 'Electrical', 'KitchenQual',
       'Functional', 'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond',
       'PavedDrive', 'SaleType', 'SaleCondition']

# -----first we showing all the categories
cat_count=df['MSZoning'].value_counts()
# print(cat1)
# ----than we showing the shape of the dataFrame and we just show the rows of the data Frame----0 means rows
cat_shape=df.shape[0]
# print(cat_shape)
# -----now we want to percentage of the categorical dataFrame
cat_per=cat_count/cat_shape*100
# print(cat_per)

# -----Now we count the percentage of after the droping empty values
drop_cat_count=df3_drop_rows['MSZoning'].value_counts()
drop_cat_shape=df3_drop_rows.shape[0]
# ----Now counts the percentage of after the droped empty values
drop_cat_per=drop_cat_count/drop_cat_shape*100
# print(drop_cat_per)

# Now we compares that using pandas function before droping empty values and after droping empty values
compare=pd.concat([cat_per,drop_cat_per],axis=1,keys=['MSZoning_org','MSZoning_clean'])
# print(compare)

# ------now we showing the data in function which we creat
def compares(val):
    # ----we someup all the values in one value due to good practice
    return pd.concat([df[val].value_counts()/df.shape[0]*100,df3_drop_rows[val].value_counts()/df3_drop_rows.shape[0]*100],axis=1,keys=[val+'_org',val+'_clear'])
# value='SaleType'
# print(compares(value))
# for item ,items in enumerate(categorical):
#     value1=items
#     item=item+1
#     print(compares(value1))

# plooting the graph of the categorical
plt.pie(cat_count)
plt.show()




