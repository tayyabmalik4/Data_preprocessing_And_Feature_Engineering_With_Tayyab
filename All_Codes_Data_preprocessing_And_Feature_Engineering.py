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


# (01)*********************************Introduction of Data Preprocessing using python************************************


# ***********(1)-What is Data Preprocessing?
# -----Data-----Definition------In computing, data is information that has been translated into a form that is efficient for movement or processing.
# -----Data-----Data is anything like(Text,images,vedios,audios etc).
# -------Definition of Data Preprocessing----Data Preprocessing is a process to convert raw data into meaningful data using different techniques.


# *********(2)-Why is Data Processing Importance?
# ----Achully Data in the real world is raw-data(dirty-data)
# ----Clearing Garbig Values----we cleaning the raw-data to meaning-full data for using to machine leaning we clean the garbeg like--- (Incomplete Data, Noisy,Inconsistend(reverse),Duplicate)
# ----converting Quality Data-----we converting the raw-data to meaning-ful data like ---(Accuracy,Completeness,Consistency,Believability,Interpretability)
# -----Build-Machine-Learning-Model-----Machine Learning Algorithms follow the rule (learn like kids)
# -----quit(GIGO----Garbage In Garbage out)


# ***********Steps to Increase the quality of data and converding the row-Data to meaning-ful Data
# ----Data Cleaning
# ----Data Integration
# ----Data Reduction
# ----Data Tranformation
# ----Data Discretization 

# ********Data Cleaning
# ----Definition---Data Cleaning means fill in missing Values,smooth, out noise while identifying outliers and correct inconsistancies in the data

# ******Data Integration
# ----Definition---Data Integration is a technique to merges data from multiple sources into a coherent data store, such as a data warehouse.

# ******Data Reduction
# ----Definition---Data Reduction is a technique use to reduce the data size by aggregating, eliminating redundant features, or clustering for instance.

# ******Data Transformation
# /////we convered the data is one formed to different many others formated like converting to pieces
# ----Definition---Data Transformation means data are transformed or consolidated into forms appropriate for ML model training, such as normalization, may be applied where dara are scaled to fall within a smaller range likem -1.0 to 1.0
# ////we transform the data using these functions-----(Aggregation, Feature type conversion, Normalization, Attribute/feature construction)

# ******Data Discretization
# ----Definition---Data Discretization technique transforms numeric data by mapping values to interval or concept labels.
# ----It can be used to reduce the number of values for a given continuous attribute by dividing the range of the attribute into intervals
# ----Discretization techniques include---(Bining, Histogram analysis, Cluster analysis, Decision-tree analysis, Correlation analysis)
# ----Example----Age(1,2,3,4,5,6,7,8,9) dividing and pieces (1-3,4-6, 7-9)


# *****************Data preprocessing lies these libraies and study others
# /////libraies
# ---(1)-numpy
# ---(2)-pandas
# ---(3)-matplotlib
# ---(4)-seaborn
# ---(5)-scikit-learn

# /////study and knowledge
# ----(1)---Mathematics
# ----(2)---Statistics
# ----(3)---Probability
# ----(4)---Calculus
# ----(5)---Linear Algebra


# (08)**********************Handling Missing Values for using Deleting technique in Data Preprocessing and Feature Engineering*******************************

df=pd.read_csv('datasets/Housing Price/train.csv')
print(df)
print(df.shape)
print(df.head(6))
print(df.tail(6))
pd.set_option('display.max_columns',None)
pd.set_option('display.max_rows',None)
print(df.tail())
print(df.info())
print(df.isnull().sum())
plt.figure(figsize=(13,13))
sns.heatmap(df.isnull())
plt.show()
percentage=(df.isnull().sum()/df.shape[0])*100
print(percentage)
drop_columns=percentage[percentage>17].keys()
print(drop_columns)
df2_drop_clm=df.drop(columns=drop_columns)
print(df2_drop_clm)
print(df2_drop_clm.shape)
sns.heatmap(df2_drop_clm.isnull())
plt.show()
df3_drop_rows=df2_drop_clm.dropna() 
print(df3_drop_rows)
print(df3_drop_rows.shape)
plt.figure(figsize=(13,13))
sns.heatmap(df3_drop_rows.isnull())
plt.show()
print(df3_drop_rows.isnull().sum())
numaric=df3_drop_rows.select_dtypes(include=['int64','float64']).columns
print(numaric)
sns.distplot(df['MSSubClass'])
sns.distplot(df3_drop_rows['MSSubClass'])
sns.distplot(df['MSSubClass'])
sns.distplot(df3_drop_rows['MSSubClass'])
plt.show()
num_val=['MSSubClass', 'LotArea', 'OverallQual', 'OverallCond',
       'YearBuilt', 'YearRemodAdd', 'MasVnrArea', 'BsmtFinSF1', 'BsmtFinSF2',
       'BsmtUnfSF', 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 'LowQualFinSF',
       'GrLivArea', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath',
       'BedroomAbvGr', 'KitchenAbvGr', 'TotRmsAbvGrd', 'Fireplaces',
       'GarageYrBlt', 'GarageCars', 'GarageArea', 'WoodDeckSF', 'OpenPorchSF',
       'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea', 'MiscVal',
       'MoSold', 'YrSold', 'SalePrice']
plt.figure(figsize=(13,13))
for item,items in enumerate(num_val):
    plt.subplot(9,4,item+1)
    sns.distplot(df[items],bins=20)
    sns.distplot(df3_drop_rows[items],bins=20)
plt.show()
cat_df=df3_drop_rows.select_dtypes('object').columns
print(cat_df)
categorical=['MSZoning', 'Street', 'LotShape', 'LandContour', 'Utilities',
       'LotConfig', 'LandSlope', 'Neighborhood', 'Condition1', 'Condition2',
       'BldgType', 'HouseStyle', 'RoofStyle', 'RoofMatl', 'Exterior1st',
       'Exterior2nd', 'MasVnrType', 'ExterQual', 'ExterCond', 'Foundation',
       'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2',
       'Heating', 'HeatingQC', 'CentralAir', 'Electrical', 'KitchenQual',
       'Functional', 'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond',
       'PavedDrive', 'SaleType', 'SaleCondition']
cat_count=df['MSZoning'].value_counts()
print(cat_count)
cat_shape=df.shape[0]
print(cat_shape)
cat_per=cat_count/cat_shape*100
print(cat_per)
drop_cat_count=df3_drop_rows['MSZoning'].value_counts()
drop_cat_shape=df3_drop_rows.shape[0]
drop_cat_per=drop_cat_count/drop_cat_shape*100
print(drop_cat_per)
compare=pd.concat([cat_per,drop_cat_per],axis=1,keys=['MSZoning_org','MSZoning_clean'])
print(compare)
def compares(val):
    return pd.concat([df[val].value_counts()/df.shape[0]*100,df3_drop_rows[val].value_counts()/df3_drop_rows.shape[0]*100],axis=1,keys=[val+'_org',val+'_clear'])
value='SaleType'
print(compares(value))
for item ,items in enumerate(categorical):
    value1=items
    item=item+1
    print(compares(value1))
plt.pie(cat_count)
plt.show()


# (09)*********************************Fill the Values technique in Handling Missing Values using Data Preprocessing and Feature Engineering******************

dataset_link="https://drive.google.com/file/d/1dGgHEwbOnBDgBpP98QmiAZO9rKcDm2tA/view?usp=sharing"
linked='https://drive.google.com/uc?export=download&id='+dataset_link.split('/')[-2]
df1=pd.read_csv(linked)
pd.set_option('display.max_columns',None)
pd.set_option('display.max_rows',None)
print(df1.head())
print(df1.shape)
print(df1.info())
print(df1.isnull().sum())
per=(df1.isnull().sum()/df1.shape[0])*100
print(per)
drop_20_per=per[per>20].keys()
print(drop_20_per)
df1_drop_col=df1.drop(columns=drop_20_per)
print(df1_drop_col.shape)
df1_num=df1_drop_col.select_dtypes(include=['int64','float64'])
print(df1_num.head())
sns.heatmap(df1_num.isnull())
plt.show()
print(df1_num.isnull().any(axis=1))
print(df1_num.isnull().sum())
var=[]
for var in df1_num.columns:
    if df1_num[var].isnull().sum()>0:
        print(var)
num_count_emp=[var for var in df1_num.columns if df1_num[var].isnull().sum()>0]
print(num_count_emp)
plt.figure(figsize=(13,13))
sns.set()
for i,var in enumerate(num_count_emp):
    plt.subplot(2,2,i+1)
    sns.distplot(df1_num[var],bins=20)
plt.show()
df2_num_mean_fill=df1_num.fillna(df1_num.mean())
print(df2_num_mean_fill.isnull().sum().sum())
df3_num_median_fill=df1_num.fillna(df1_num.median())
print(df3_num_median_fill.isnull().sum().sum())
plt.figure(figsize=(13,13))
sns.set()
for i ,var in enumerate(num_count_emp):
    plt.subplot(2,2,i+1)
    sns.distplot(df1_num[var],bins=20,label="original",kde_kws={'linewidth':8,'color':'r'})
    sns.distplot(df2_num_mean_fill[var],bins=20,label="Mean",kde_kws={'linewidth':5,'color':'g'})
    plt.legend()
plt.show()
plt.figure(figsize=(13,13))
sns.set()
for i,var in enumerate(num_count_emp):
    plt.subplot(2,2,i+1)
    sns.distplot(df1_num[var],hist=False,bins=20,label="original",kde_kws={'linewidth':8,'color':'r'})
    sns.distplot(df2_num_mean_fill[var],hist=False,bins=20,label="Mean",kde_kws={'linewidth':5,'color':'g'})
    sns.distplot(df3_num_median_fill[var],hist=False,bins=20,label= "Median",kde_kws={'linewidth':3,'color':'k'})
    plt.legend()
plt.show()
for i,var in enumerate(num_count_emp):
    plt.figure(figsize=(13,13))
    plt.subplot(9,3,1)
    sns.boxplot(df1[var])
    plt.subplot(3,1,2)
    sns.boxplot(df2_num_mean_fill[var])
    plt.subplot(3,1,3)
    sns.boxplot(df3_num_median_fill[var])
    plt.legend()
plt.show()
df_concat=pd.concat([df1_num[num_count_emp],df2_num_mean_fill[num_count_emp],df2_num_mean_fill[num_count_emp]], axis=1)
print(df_concat[df_concat.isnull().any(axis=1)])



# (10)************************Fill the Values by class technique of missing values in Data Preprocessing and Feature Engineering**********************




file_link="https://drive.google.com/file/d/1dGgHEwbOnBDgBpP98QmiAZO9rKcDm2tA/view?usp=sharing"
linked="https://drive.google.com/uc?export=download&id="+file_link.split('/')[-2]
df=pd.read_csv(linked)
print(df.head())
print(df.shape)
pd.set_option('display.max_columns',None)
pd.set_option('display.max_rows',None)
print(df.head())
print(df.isnull().sum())
per=df.isnull().sum()/df.shape[0]*100
print(per)
df1_show_clm=per[per>20].keys()
print(df1_show_clm)
df2_drop_clm=df.drop(columns=df1_show_clm)
print(df2_drop_clm.shape)
df3_num=df2_drop_clm.select_dtypes(include=['int64','float64'])
print(df3_num.info())
print(df3_num.isnull().sum())
df4_num_emp=[var for var in df3_num.columns if df3_num[var].isnull().sum()>0]
print(df4_num_emp)
print(df3_num[df4_num_emp][df3_num[df4_num_emp].isnull().any(axis=1)])
print(df['LotConfig'].unique())
locations=df[df.loc[:,'LotConfig']=='Inside']["LotFrontage"]
print(locations)
replac=locations.replace(np.nan,locations.mean())
print(replac)
df_copy=df.copy()
for var_class in df['LotConfig'].unique():
    df_copy.update(df[df.loc[:,'LotConfig']==var_class]['LotFrontage'].replace(np.nan,df[df.loc[:,'LotConfig']==var_class]['LotFrontage'].mean()))
print(df_copy.isnull().sum())
num_vars_miss=['LotFrontage', 'MasVnrArea', 'GarageYrBlt']
cat_vars=['LotConfig','MasVnrType','GarageType']
for cat_var,num_var_miss in zip(cat_vars,num_vars_miss):
    for var_class in df[cat_var].unique():
        df_copy.update(df[df.loc[:,cat_var]==var_class][num_var_miss].replace(np.nan,df[df.loc[:,cat_var]==var_class][num_var_miss].mean()))
print(df_copy[num_vars_miss].isnull().sum())
print(df_copy[df_copy[['MasVnrType']].isnull().any(axis=1)])
num_vars_miss=['LotFrontage', 'MasVnrArea', 'GarageYrBlt']
cat_vars=['LotConfig','Exterior2nd','KitchenQual'] 
for cat_var,num_var_miss in zip(cat_vars,num_vars_miss):
    for var_class in df[cat_var].unique():
        df_copy.update(df[df.loc[:,cat_var]==var_class][num_var_miss].replace(np.nan,df[df.loc[:,cat_var]==var_class][num_var_miss].mean()))
print(df_copy[num_vars_miss].isnull().sum())
df_copy_median=df.copy()
for cat_var,num_var_miss in zip(cat_vars,num_vars_miss):
    for var_class in df[cat_var].unique():
        df_copy_median.update(df[df.loc[:,cat_var]==var_class][num_var_miss].replace(np.nan,df[df.loc[:,cat_var]==var_class][num_var_miss].median()))
print(df_copy_median[num_vars_miss].isnull().sum())
plt.figure(figsize=(13,13))
sns.set()
for i,var in enumerate(num_vars_miss):
    plt.subplot(2,2,i+1)
    sns.distplot(df[var],bins=20,kde_kws={'linewidth':8,'color':'r'},label='Original')
    sns.distplot(df_copy[var],bins=20,kde_kws={'linewidth':5,'color':'g'},label='Mean')
    sns.distplot(df_copy_median[var],bins=20,kde_kws={'linewidth':3,'color':'k'},label='Median')
    plt.legend()
plt.show()
for i,var in enumerate(num_vars_miss):
    plt.subplot(3,1,1)
    sns.boxplot(df[var])
    plt.subplot(3,1,2)
    sns.boxplot(df_copy[var])
    plt.subplot(3,1,3)
    sns.boxplot(df_copy_median[var])
    plt.show()


# (11)*******************************Fill Values by helping Mode function technique part 05 of the Cleaning Missing Values in Data Preprocessing and Feature Engineering***************************

df=pd.read_csv('datasets/Housing Price/train.csv')
print(df.head())
pd.set_option('display.max_columns',None)
pd.set_option('display.max_rows',None)
df2_cat=df.select_dtypes(include='object')
print(df2_cat.shape)
print(df2_cat.info())
per=df2_cat.isnull().sum()/df2_cat.shape[0]*100
print(per)
per2=df2_cat.isnull().mean()*100
print(per2)
df3_drop_20per=per[per>20].keys()
print(df3_drop_20per)
df4_drop_clm=df2_cat.drop(columns=df3_drop_20per,inplace=True)
print(df4_drop_clm.shape)
df4_drop_clm_per=df2_cat.isnull().sum()/df2_cat.shape[0]*100
print(df4_drop_clm_per)
df4_drop_clm_per2=df4_drop_clm.isnull().mean()*100
print(df4_drop_clm_per2)
df5_cat_0per=df4_drop_clm_per[df4_drop_clm_per>0].keys()
print(df5_cat_0per)
fill_nan_df5=df2_cat['MasVnrType'].fillna("Missing")
print(df5_cat_0per)
print(fill_nan_df5)
cat_fill_mode=df2_cat['MasVnrType'].mode()
checking_count=df2_cat['MasVnrType'].value_counts()
print(checking_count)
cat_fill_nan_mode=df2_cat['MasVnrType'].fillna(df2_cat['MasVnrType'].mode()[0])
for var in df5_cat_0per:
    df2_cat[var].fillna(df2_cat[var].mode()[0],inplace=True)
    print(var +"="+ df2_cat[var].mode()[0])
print(df2_cat.isnull().sum())
plt.figure(figsize=(13,13))
sns.set()
for i,var in enumerate(df5_cat_0per):
    plt.subplot(4,3,i+1)
    plt.hist(df2_cat[var],label='Mode')
    plt.hist(df[var].dropna(),label='Real')
    plt.legend()
plt.show()
df.update(df2_cat)
df.drop(columns=df3_drop_20per,inplace=True)
updated=df.select_dtypes(include='object').isnull().sum()
print(updated)


# (12)*************************Fill values using sklearn library technique in the DataFrame Handling Missing Values in Data Preprocessing and Feature Engineering***********************************

train=pd.read_csv(r'datasets/Housing Price/train.csv')
test=pd.read_csv(r'datasets/Housing Price/test.csv')
print(train.head())
print(test.head())
print('shape of train df =',train.shape)
print('shape of test df =',test.shape)
X_train=train.drop(columns='SalePrice')
y_train=train['SalePrice']
print('shape of X_train df =',X_train.shape)
print('shape of y_train df =',y_train.shape)
num_vars=X_train.select_dtypes(include=['int64','float64']).columns
print(num_vars)
print(X_train[num_vars].isnull().sum())
imputer_mean=SimpleImputer(strategy='mean')
imputer_mean2=SimpleImputer(strategy='constant',fill_values='Missing')
sk_fit=imputer_mean.fit(X_train[num_vars])
print(sk_fit)
sk_stat=imputer_mean.statistics_
print(sk_stat)
sk_trans=imputer_mean.transform(X_train[num_vars])
print(sk_trans)
X_train[num_vars]=imputer_mean.transform(X_train[num_vars])
test[num_vars]=imputer_mean.transform(test[num_vars])
print(X_train[num_vars].isnull().sum())
print(test[num_vars].isnull().sum())
cat_vars=X_train.select_dtypes(include='object').columns
print(cat_vars )
impute_mode=SimpleImputer(strategy='most_frequent')
sk_fit_cat=impute_mode.fit(X_train[cat_vars])
print(impute_mode.statistics_)
X_train[cat_vars]=impute_mode.transform(X_train[cat_vars])
test[cat_vars]=impute_mode.transform(test[cat_vars])
print(X_train[cat_vars].isnull().sum())
print(test[cat_vars].isnull().sum())


# (13)-*************************Fill Missing Values using sklearn library Hanling Missing Values in Data Preprocessing and Feature Engineering**********************************

train=pd.read_csv(r"datasets/Housing Price/train.csv")
test=pd.read_csv(r"datasets/Housing Price/test.csv")
print('shape of train df =',train.shape)
print('shape of test df = ',test.shape)
X_train=train.drop(columns='SalePrice',axis=1)
y_train=train["SalePrice"]
X_test=test.copy()
print("shape of X_train =",X_train.shape)
print("shape of y_train = ",y_train.shape)
print("shape of X_test = ",X_test.shape)
isnull_values=X_train.isnull().sum()
print(isnull_values)
num_vars=X_train.select_dtypes(include=['int64','float64']).columns
num_vars_miss=[var for var in num_vars if isnull_values[var]>0]
print(num_vars_miss)
num_vars_emp=['LotFrontage', 'MasVnrArea', 'GarageYrBlt']
cat_vars=X_train.select_dtypes(include='object').columns
cat_vars_miss=[var for var in cat_vars if isnull_values[var]>0]
print(cat_vars_miss)
cat_vars_emp=['Alley', 'MasVnrType', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'Electrical', 'FireplaceQu', 'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond', 'PoolQC', 'Fence', 'MiscFeature']
num_var_mean=['LotFrontage']
num_var_median=['MasVnrArea','GarageYrBlt']
num_var_mode=['Alley', 'MasVnrType', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'Electrical', 'FireplaceQu']
num_var_constant=['GarageType', 'GarageFinish', 'GarageQual', 'GarageCond', 'PoolQC', 'Fence', 'MiscFeature']
num_var_mean_imputer=Pipeline(steps=[('imputer',SimpleImputer(strategy='mean'))])
num_var_median_imputer=Pipeline(steps=[("imputer",SimpleImputer(strategy="median"))])
num_var_mode_imputer=Pipeline(steps=[("imputer",SimpleImputer(strategy="most_frequent"))])
num_var_constant_imputer=Pipeline(steps=[("imputer",SimpleImputer(strategy="constant",fill_value="Missing"))])
preprocesser=ColumnTransformer(transformers=[('mean_imputer',num_var_mean_imputer,num_var_mean),("median_imputer",num_var_median_imputer,num_var_median),("mode_imputer",num_var_mode_imputer,num_var_mode),("constant_imputer",num_var_constant_imputer,num_var_constant)])
df_fit=preprocesser.fit(X_train)
print(df_fit)
values_checked=preprocesser.named_transformers_["mean_imputer"].named_steps["imputer"].statistics_ 
print(values_checked)
check_mode=preprocesser.named_transformers_["mode_imputer"].named_steps['imputer'].statistics_
print(check_mode)
print(train['LotFrontage'].mean())
X_train_clean=preprocesser.transform(X_train)
X_test_clean=preprocesser.transform(X_test)
print(X_train_clean)
print(preprocesser.transformers_)
X_train_clean_miss_var=pd.DataFrame(X_train_clean,columns=num_var_mean+num_var_median+num_var_mode+num_var_constant)
print(X_train_clean_miss_var.head())
print(X_train_clean_miss_var.shape)
print(train["Alley"].value_counts())
print(X_train_clean_miss_var["Alley"].value_counts())
print(X_train_clean_miss_var["MiscFeature"].value_counts())


# (14)**********************One Hot Encoding Dummy variables in Categorical variables in data preprocessing and Feature Engineering**************************



df=pd.read_csv('datasets/hotal dataset/tips.csv')
print(df.head())
df1=df.drop(columns=['Payer Name','CC Number','Payment ID'],inplace=True)
df2=pd.get_dummies(df)
print(df2.head())
print(df2.keys())
df3=pd.get_dummies(df,drop_first=True)
print(df3)
oh_enc=OneHotEncoder(sparse=False)
oh_enc_fit=oh_enc.fit_transform(df[['sex','smoker','day','time']])
converter=pd.DataFrame(oh_enc_fit,columns=['sex_Female','sex_Male', 'smoker_No', 'smoker_Yes', 'day_Fri', 'day_Sat', 'day_Sun','day_Thur', 'time_Dinner', 'time_Lunch'])
print(converter)


# (15)****************Label Encoding and also learn Ordinal Encoding using sklearnn library in Data Preprocessing and Feature Engineering******************

df=pd.read_csv(r"datasets/Housing Price/train.csv")
pd.set_option('display.max_columns',None)
pd.set_option('display.max_rows',None)
print(df.head())
df2=df[['KitchenQual','BldgType']]
le=LabelEncoder()
le1=le.fit_transform(df2["BldgType"])
print(le1)
df2['BldgType_L_enc']=le.fit_transform(df2['BldgType'])
print(df2)
print(df['BldgType'].value_counts())
print(df['KitchenQual'].value_counts())
order_label={'Ex':4,'Gd':3,'TA':2,'Fa':1}
df2['KitchenQual_org_enc']=df2['KitchenQual'].map(order_label)
print(df2)


# (17)*******************Feature Scaling in Standardization And Normalization in Data Preprocessing and Feature Engineering***********************



df=sns.load_dataset('titanic')
print(df.head())
df2=df[['survived','pclass','age','parch']]
print(df2.head())
df3=df2.fillna(df2.mean())
X=df3.drop('survived',axis=1)
y=df3['survived']
print('shape of X is:',X.shape)
print('shape of y is:',y.shape)
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=51)
print('Shape of X_train is: ',X_train.shape)
print('shape of y_train is: ',y_train.shape)
print('shape of X_test is: ',X_test.shape)
print('shape of y_test is: ',y_test.shape)
sc=StandardScaler()
sc_fit=sc.fit(X_train)
print(sc_fit)
print(sc.mean_)
print(sc.scale_)
print(X_train.describe())
X_train=sc.transform(X_train)
X_test=sc.transform(X_test)
print(X_train)
print(X_test)
X_train_sc_con=pd.DataFrame(X_train,columns=['pclass','age','parch'])
X_test_sc_con=pd.DataFrame(X_test,columns=['pclass','age','parch'])
print(X_train_sc_con)
print(X_test_sc_con)
print(X_train_sc_con.describe())
print(X_train_sc_con.describe().round(2))
mmc=MinMaxScaler()
X_train_mmc=mmc.fit(X_train).transform(X_train)
X_test_mmc=mmc.fit(X_test).transform(X_test)
print(X_train_mmc)
print(X_test_mmc)
X_train_mmc_tran=mmc.transform(X_train_mmc)
X_test_mmc_tran=mmc.transform(X_test_mmc)
print(X_train_mmc_tran)
print(X_test_mmc_tran)
X_train_mmc_df=pd.DataFrame(X_train_mmc,columns=['pclass','age','parch'])
X_test_mmc_df=pd.DataFrame(X_test_mmc,columns=['pclass','age','parch'])
print(X_train_mmc_df)
print(X_test_mmc_df)
print(X_train_mmc_df.describe())
print(X_train_mmc_df.describe())











