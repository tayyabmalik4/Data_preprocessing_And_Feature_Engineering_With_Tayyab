# (16)***************Feature Scaling Technique in data preprocessing and Feature Engineering*************************

# -----Discuss About-----
# ---(1)-What is Feature
# ---(2)-What us Scaling
# ---(3)-What is Feature Engineering
# ---(4)-Why & use Feature Engineering
# ---(5)-Types of Feature Engineering
# ---(6)-Advantages
# ---(7)-Disadvantages
# ---(8)-Algorithm Support Feature Engineering



# **************What is Feature
# -----Eveery column in dataset is like a feature ----for example----Male Hight in Feet

# **************What is Scale
# -----scale means the numaric number which exists the minmum and maximum numbers of the belowing columns----for example Male Hight in Feet is a column of dataset and this is a Feature----and maximum feet exists in the column is 8.5 so the scale of Male Hight in Feet is 0 to 8.5

# **************What is Feature Scaling  
# ---Feature Scaling is a method to scale numeric features in the same scale or range(like: -1 to 1, 0 to 1)
# ----This is last step involved in Data Preprocessing and before ML model training
# ----It is also called as data normalization.
# ----We apply Feature Scaling on independent variables.
# ----We fit feature scaling with train data and transform on train and test data


# *********Why Feature Scaling?
# ----The scale of raw features is different according to its units.
# ----Machine Learing algorithms can't understand features unites,understand only numbers.
# ----For Example----if Hight 140cm and 8.2 feet
# ----We understand it very easily that the 8.2 feet is greater than 140cm
# ----But ML Algorithms understan numbers then 140>8.2
# -----So we apply the feature Scaling to avoid this mistake


# *********Which ML Algorithms RequiredFeature Scaling?
# ----Those Algorithms Calculate Distance
# ---(1)-K-Nearest Neighbors(KNN)
# ---(2)-K-Means
# ---(3)-Suppport Vectory Machine(SVM)
# ---(4)---Principal Component Analysis(PCA)
# ---(5)---Linear Discriminant Analysis
# ----Distance Formula----

# *******Gradient Descent Based Algorithms
# ---Linear Regreesion
# ---Logistic Regression
# ---Neural Network

# *******Which algorithms not Required Feature Scaling----These Algorithms have not impacted on Feature Engineering
# ---Decision Tree, Random Forest, XGBoost

# **************Advantages***********************
# -----The accuracy is increse many times of Machine Learning
# -----to increasing the speed of Machine Learing


# ******Types of Feature Scaling
# ----(1)-Min Max Scaler------Maximum used
# ----(2)-Standard Scaler-----Maximum Used
# ----(3)-Max Abs Scale
# ----(4)-Robust Scaler
# ----(5)-Quantile Transformer Scaler
# ----(6)-Power Transformer Scaler
# ----(7)-Unit Vector Scaler


# ----------------------------------END-------------------------------