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

