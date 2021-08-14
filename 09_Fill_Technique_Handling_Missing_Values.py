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
print(df1.head())