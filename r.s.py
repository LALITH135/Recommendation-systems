  # -*- coding: utf-8 -*-
"""
Created on Wed May 20 16:41:09 2022

@author: lalith kumar
"""
# Building a recommenndation system by using cosine simillarties score.

# import dataset.
import pandas as pd
import numpy as np

df = pd.read_csv('E:\\data science\\ASSIGNMENTS\\ASSIGNMENTS\\Recommendation systems\\book.csv',encoding='latin-1')
df.shape

df.head()
df.describe()
df.info()

pd.set_option('display.max.columns',None)
df.rename(columns={'User.ID':'user_id','Book.Title':'book_title','Book.Rating':'book_rating'}, inplace=True)

df.sort_values('user_id')

df.duplicated('user_id').sum()

#number of unique users in the dataset
len(df)
len(df.user_id.unique())


# histogram plot

df['book_rating'].value_counts()
df['book_rating'].hist()

len(df.book_title.unique())
df.book_title.value_counts()

user_df = df.pivot_table(index='user_id',
                                 columns='book_title',
                                 values='book_rating')

user_df
type(user_df)
user_df.shape

user_df.iloc[:,0].describe()


user_df.iloc[0]
user_df.iloc[200]
list(user_df)

#Impute those NaNs with 0 values
user_df.fillna(0, inplace=True)

user_df.shape

# from scipy.spatial.distance import cosine correlation
# Calculating Cosine Similarity between Users
from sklearn.metrics import pairwise_distances
user_sim = 1 - pairwise_distances( user_df.values,metric='cosine')

#user_sim = 1 - pairwise_distances( user_df.values,metric='correlation')

user_sim.shape

#Store the results in a dataframe
user_sim_df = pd.DataFrame(user_sim)

#Set the index and column names to user ids 
user_sim_df.index   = df.user_id.unique()
user_sim_df.columns = df.user_id.unique()

# CHECKING CORRELATION
user_sim_df.iloc[0:5,0:5]

user_sim_df.iloc[:,5].describe()

# CONVERT DIAGONAL VALUES AS ZERO'S
np.fill_diagonal(user_sim, 0)
user_sim_df.iloc[0:5, 0:5]

# Most Similar Users
user_sim_df.max()
    
# highest correlation of first 5
user_sim_df.idxmax(axis=1)[0:5]

# checking the similarities between two user_id's.

df[(df['user_id']==276729) | (df['user_id']==276726)]

# checking individual user_id.
user_276729=df[df['user_id']==276729]
user_276726=df[df['user_id']==276726]

df[(df['user_id']==276736) | (df['user_id']==276726)]

user_276736=df[df['user_id']==276736]
user_276726=df[df['user_id']==276726]

pd.merge(user_276729,user_276726,on='book_title',how='inner')
pd.merge(user_276736,user_276726,on='book_title',how='outer')

#=======================================================================











