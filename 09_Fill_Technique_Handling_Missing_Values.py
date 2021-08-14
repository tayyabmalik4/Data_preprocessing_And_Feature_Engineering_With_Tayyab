# (09)*********************************Fill the Values technique in Handling Missing Values using Data Preprocessing and Feature Engineering******************

# ----About-----
# (1)-Fill missing values manually
# (2)-Global constant
# (3)-Measure of central tendency(Mean,Median)

# ----When the values are numarical we use Mean and Median
# ----When the values are categorical we use  Mode

# ----importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ----imporing csv file----this is the simple way to importing file
# df=pd.read_csv('datasets/Housing Price/train.csv')
# ----but if the file is place of the goggle drive than we use following steps
# ----we just copy and past the link and than
dataset_link="https://drive.google.com/file/d/1dGgHEwbOnBDgBpP98QmiAZO9rKcDm2tA/view?usp=sharing"
linked='https://drive.google.com/uc?export=download&id='+dataset_link.split('/')[-2]
df1=pd.read_csv(linked)
# ----display all columns and rows
pd.set_option('display.max_columns',None)
pd.set_option('display.max_rows',None)
# print(df1.head())

# ----to display the just total columns and rows
# print(df1.shape)

# ----To check the information of the data set
# print(df1.info())

# ----to printing the information of null values 
# print(df1.isnull().sum())

# ----to printing the percentage of dataframe null values
per=(df1.isnull().sum()/df1.shape[0])*100
# print(per)

# ----The rule is droping these columns who is empty values are presents 20% but Now this case ... 
drop_20_per=per[per>20].keys()
print(drop_20_per)
# ----droping these columns
df1_drop_col=df1.drop(columns=drop_20_per)
print(df1_drop_col.shape)

# ----now printout these columns who the datatype is numaric
df1_num=df1_drop_col.select_dtypes(include=['int64','float64'])
# print(df1_num.head())

# -----Now check that the empty values in the numaric  variables
# sns.heatmap(df1_num.isnull())
# plt.show()

# -----check the null values in numaric variables
# print(df1_num.isnull().any(axis=1))

# ----printing the null valueus of numaric variables
# print(df1_num.isnull().sum())

# ----printing whose variables who is null values are present using for loop
# ----1st method...
# var=[]
# for var in df1_num.columns:
#     if df1_num[var].isnull().sum()>0:
#         print(var)
# ----2nd method...
num_count_emp=[var for var in df1_num.columns if df1_num[var].isnull().sum()>0]
# print(num_count_emp)


# ----when our numarical values are present randomly than we use mean median and mode
# ----When our numaric values are present seqauntly than other functions are use

# -----checking the values as a diagram
# plt.figure(figsize=(13,13))
# sns.set()
# for i,var in enumerate(num_count_emp):
#     plt.subplot(2,2,i+1)
#     sns.distplot(df1_num[var],bins=20)
# plt.show()

# -----fill values using mean function
df2_num_mean_fill=df1_num.fillna(df1_num.mean())
# print(df2_num_mean_fill.isnull().sum().sum())

# ----fill values using median function
df3_num_median_fill=df1_num.fillna(df1_num.median())
# print(df3_num_median_fill.isnull().sum().sum())


# ----Now Evaluate that the the real dataframe and existance datafram using seaborn library
# plt.figure(figsize=(13,13))
# sns.set()
# for i ,var in enumerate(num_count_emp):
#     plt.subplot(2,2,i+1)
#     sns.distplot(df1_num[var],bins=20,label="original",kde_kws={'linewidth':8,'color':'r'})
#     sns.distplot(df2_num_mean_fill[var],bins=20,label="Mean",kde_kws={'linewidth':5,'color':'g'})
#     plt.legend()
# plt.show()

# ----Now we creat graphs of median, mean and real values
# plt.figure(figsize=(13,13))
# sns.set()
# for i,var in enumerate(num_count_emp):
#     plt.subplot(2,2,i+1)
#     sns.distplot(df1_num[var],hist=False,bins=20,label="original",kde_kws={'linewidth':8,'color':'r'})
#     sns.distplot(df2_num_mean_fill[var],hist=False,bins=20,label="Mean",kde_kws={'linewidth':5,'color':'g'})
#     sns.distplot(df3_num_median_fill[var],hist=False,bins=20,label= "Median",kde_kws={'linewidth':3,'color':'k'})
#     plt.legend()
# plt.show()


# -----plotting the graph using boxplot
# for i,var in enumerate(num_count_emp):
#     plt.figure(figsize=(13,13))
#     plt.subplot(9,3,1)
#     sns.boxplot(df1[var])
#     plt.subplot(3,1,2)
#     sns.boxplot(df2_num_mean_fill[var])
#     plt.subplot(3,1,3)
#     sns.boxplot(df3_num_median_fill[var])
#     plt.legend()
# plt.show()

# -----Now we want to checking the deffernce and now we use concatinate function and print the values
df_concat=pd.concat([df1_num[num_count_emp],df2_num_mean_fill[num_count_emp],df2_num_mean_fill[num_count_emp]], axis=1)
print(df_concat[df_concat.isnull().any(axis=1)])
