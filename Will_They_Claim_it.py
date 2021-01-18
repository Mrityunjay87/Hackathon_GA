# -*- coding: utf-8 -*-
"""
Created on Thu Dec 17 07:34:24 2020

@author: Mrityunjay1.Pandey
"""
#Importing necessary librarires

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.preprocessing import StandardScaler
from scipy import stats
from seaborn_qqplot import pplot
#Reading Data

train_path='./train.csv'
test_path='./test.csv'

df_train=pd.read_csv(train_path)
df_test=pd.read_csv(test_path)

#Checking data


skew_check={}
numerical_cols=df_train.select_dtypes(include=['int64','float64'])
categorical_cols=df_train.select_dtypes(include='object')
numerical_cols.drop(["ID","Claim"],inplace=True,axis=1)

for i in numerical_cols.columns:
    sns.distplot(df_train[i],kde=True)
    plt.show()
    sns.scatterplot(numerical_cols[i],df_train.Claim)
    plt.show()
    #Checking Skewness of the data
    skew_check[i]=numerical_cols[i].skew()

df_train_tranformed=pd.DataFrame()
#og transforamtion didn't work as data has 0 values "RuntimeWarning: divide by zero encountered in log"    
for i in numerical_cols.columns:
   df_train_tranformed[i]=np.log(df_train[i]+1)
   sns.distplot(df_train[i],kde=True)
   plt.show()
   pplot(df_train_tranformed, x="Age", y=df_train_tranformed[i], kind='qq', height=4, aspect=2, display_kws={"identity":False, "fit":True})
   plt.show()

sns.pairplot(numerical_cols)
plt.show()
#Applying Sqrt transformation for first time didn't remove skewness completely hence we have to run it twice.
#    Checking if other transformation can work better then this.
# =============================================================================
# Tranformed_skew_check={}
# for i in numerical_cols.columns:
#     df_train[i]=np.sqrt(df_train[i])
#     sns.distplot(df_train[i],kde=True)
#     plt.show()
#     Tranformed_skew_check[i]=df_train[i].skew()
#     if Tranformed_skew_check[i]>0:
#         df_train[i]=np.sqrt(df_train[i])
#         Tranformed_skew_check[i]=df_train[i].skew()
# =============================================================================

        
#Inspecting skenewss value and plot it is clear data is skewed hence using trnasformation techniques to remove skewness

# =============================================================================
# Applying Box Cox Transformation
# =============================================================================
from scipy import stats

df_train_box_transformed=pd.DataFrame()
df_train_box_transformed['Commision (in value)'], _ = stats.boxcox(numerical_cols['Commision (in value)']+1)
prob = stats.probplot(df_train_box_transformed['Commision (in value)'], dist=stats.norm)
plt.show()


#Data Massaging


   

