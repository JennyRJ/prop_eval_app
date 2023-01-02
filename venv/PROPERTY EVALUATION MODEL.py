#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip install opendatasets')


# IMPORTING DEPENDANCIES

# In[2]:


import opendatasets as od


# In[3]:


od.download("https://www.kaggle.com/datasets/jeninepaula/apartment-prices")


# In[4]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import sklearn.datasets
from sklearn.model_selection import train_test_split
from sklearn import metrics
import pickle


# In[5]:


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[6]:


os.listdir()


# In[7]:


dataset = pd.read_csv("apartment-prices/rent_apartments.csv")


# In[8]:


#Shows the first 5 items in the dataset
dataset.head()


# In[9]:


#fetch the dimensions of Pandas and NumPy type objects in python.
dataset.shape


# In[10]:


#prints information about the DataFrame
dataset.info()


# In[11]:


#returns object containing counts of unique values
for column in dataset.columns:
    print(dataset[column].value_counts())
    print("*"*20)


# In[12]:


#returns a DataFrame object where all the values are replaced with a Boolean value True for NA (not-a -number) values, and otherwise False

dataset.isna().sum()


# In[13]:


#clean data by removing specified columns(drop agency and link)

dataset.drop(columns=["Agency","link"],inplace=True)


# In[14]:


#calculate some statistical data like percentile, mean and std of the numerical values of the Series or DataFrame
dataset.describe()


# In[15]:


#prints information about the DataFrame
dataset.info()


# In[16]:


#lets fill the missing values
dataset["Neighborhood"].value_counts()


# In[17]:


#replaces the NULL values with a specified value
dataset["Neighborhood"]=dataset["Neighborhood"].fillna("Kileleshwa, Dagoretti North")


# In[18]:


dataset["sq_mtrs"].value_counts()


# In[19]:


dataset["Bathrooms"]=dataset["Bathrooms"].fillna(dataset["Bathrooms"].median())


# In[20]:


dataset["Bedrooms"]=dataset["Bedrooms"].fillna(dataset["Bedrooms"].median())


# In[21]:


dataset["sq_mtrs"]=dataset["sq_mtrs"].fillna(dataset["sq_mtrs"].median())


# In[22]:


dataset.info()


# In[23]:


dataset["sq_mtrs"].unique()


# In[24]:


#cleaned data
dataset.head()


# In[25]:


#drop ksh in price and the comma
dataset['Price'] = dataset['Price'].str.replace('KSh','',regex=True).str.replace(',','').astype(float)


# In[26]:


dataset.head()


# In[27]:


dataset["price_per_sq_mtrs"]= dataset["Price"] / dataset["sq_mtrs"]


# In[28]:


dataset["price_per_sq_mtrs"]


# In[29]:


dataset.describe()


# In[30]:


dataset["price_per_sq_mtrs"]=dataset["price_per_sq_mtrs"].fillna(dataset["price_per_sq_mtrs"].median())


# In[31]:


dataset.info()


# In[32]:


dataset["Neighborhood"].value_counts()


# In[33]:


dataset["Neighborhood"]=dataset["Neighborhood"].apply(lambda x: x.strip())
location_count = dataset["Neighborhood"].value_counts()


# In[34]:


location_count


# In[35]:


location_count_less_10=location_count[location_count<=10]
location_count_less_10


# In[36]:


dataset["Neighborhood"].value_counts()


# In[37]:


dataset.shape


# In[38]:


dataset.drop(columns=['price_per_sq_mtrs'],inplace=True)


# In[39]:


dataset.head()


# In[40]:


dataset.to_csv("Cleaned_data.csv")


# In[41]:


X=dataset.drop(columns=['Price'])
Y=dataset['Price']


# In[42]:


X = dataset[["Bedrooms", "Bathrooms"]]
Y = dataset['Price']

# In[44]:


import seaborn as sns
import matplotlib.pyplot as plt
sns.heatmap(dataset.isnull(),yticklabels=True,cbar=True,cmap='Blues')


# SPLIT DATA INTO TRAINING DATA AND TESTING DATA

# In[45]:


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)


# In[46]:


print(X_train.shape)
print(X_test.shape)


# In[47]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.metrics import r2_score


# MODEL TRAINING

# In[48]:


model=XGBRegressor()
#training the model with x_train
model.fit(X_train, Y_train)


# EVALUATION
# Prediction on training data

# In[49]:


#accuracy for prediction on training data
training_data_prediction = model.predict(X_train)


# In[50]:


print(training_data_prediction)


# In[51]:


#R squared error
score_1 =metrics.r2_score(Y_train,training_data_prediction)


# In[52]:


#Mean absolute error
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 2)
score_2 = metrics.mean_absolute_error(Y_train,training_data_prediction)


# In[53]:


print("R squared error: ", score_1)
print("Mean absolute error: ", score_2)


# In[54]:


#If the error is closer to 0 that means our model is performing perfectly


# In[55]:


import pickle


# In[56]:

#load model regressor on the pickle file
pickle.dump(pipe, open('RegressorModel.pkl','wb'))
#load model from the disk
pipe =pickle.load(open('RegressorModel.pkl', 'rb'))

