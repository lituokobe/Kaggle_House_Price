
# coding: utf-8

# # This is the first Kaggle competition Li Kobe Kobe joins#

# # 1. Configure the libraries and import the data#

# In[1]:


import numpy as np
import pandas as pd


# In[15]:


train_df = pd.read_csv('./input/train.csv', index_col = 0)
test_df = pd.read_csv('./input/test.csv', index_col = 0)
#We don't need pandas' default column index


# ### Check the data###

# In[17]:


train_df.head()


# In[18]:


train_df.shape


# In[19]:


test_df.head()


# In[20]:


test_df.shape


# The training set and the test set have almost the same amount of data entries, while the test set doesn't have the "SalePrice" column which we need to predict.

# # 2. Data processing (Result)#

# ### First, let's check the distribution of "SalePrice" in the training set.###

# In[24]:


get_ipython().run_line_magic('matplotlib', 'inline')
prices = train_df['SalePrice']


# In[25]:


prices.hist()


# We notice that the data is not normal distributed, so we use log1p to make it normal.

# In[27]:


np.log1p(train_df['SalePrice']).hist()


# This is normalized, we are going to use it as the result of trianing set.

# In[31]:


y_train = np.log1p(train_df.pop('SalePrice'))


# We will process the features (other columns) as well. In order to make the features in training set and test set uniform, we will combine them to process together.

# In[49]:


all_df = pd.concat((train_df,test_df),axis=0)


# In[56]:


all_df.shape


# In[51]:


y_train.head()


# Everthing looks good

# # 3. Data Processing (Feature Engineering)#

# Based on the description on Kaggle, 'MSSubClass' is actually a category instead of a numeric value.

# In[52]:


all_df['MSSubClass'].dtypes


# So we need to convert these values into string.

# In[53]:


all_df['MSSubClass'] = all_df['MSSubClass'].astype(str)


# In[54]:


all_df['MSSubClass'].value_counts()


# There are other features that are logistic values, we would like to convert them into binary format.

# In[58]:


all_dummy_df = pd.get_dummies(all_df)
all_dummy_df.head()


# In[95]:


all_dummy_df.shape


# Then, let's check the missing values.

# In[64]:


all_dummy_df.isnull().sum().sort_values(ascending=False).head(12)


# We see the column of "LotFrontage" has the most missing values, and we are going to fill them with average values.

# In[65]:


all_dummy_df = all_dummy_df.fillna(all_dummy_df.mean())


# In[69]:


all_dummy_df.isnull().sum().sum()


# Then, we pick those columns that are not binary values (numeric values), to normalize them.

# In[98]:


filter1 = all_dummy_df.dtypes == 'int64'
filter2 = all_dummy_df.dtypes == 'float64'
numeric_col = all_dummy_df.columns[filter1 | filter2]
numeric_col


# In[105]:


numeric_col_mean = all_dummy_df.loc[:,numeric_col].mean()
numeric_col_std = all_dummy_df.loc[:,numeric_col].std()
all_dummy_df.loc[:,numeric_col] = (all_dummy_df.loc[:,numeric_col] - numeric_col_mean)/ numeric_col_std


# In[106]:


all_dummy_df.head()


# The data processing has finished.

# # 4. Build the Model #

# In[111]:


dummy_train_df = all_dummy_df.loc[train_df.index]
dummy_test_df = all_dummy_df.loc[test_df.index]


# In[112]:


dummy_train_df.shape,dummy_test_df.shape


# In[118]:


X_train = dummy_train_df.values
X_test = dummy_test_df.values


# ## 4.1 Ridge Regression##

# In[127]:


from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt


# In[138]:


alphas = np.logspace(-3,2,50)#from 0.001 to 100 to evenly sample 50 values
test_scores = []
for alpha in alphas:
    clf = Ridge(alpha)
    test_score = np.sqrt(-cross_val_score(clf,X_train,y_train,cv=10,scoring='neg_mean_squared_error'))
    test_scores.append(np.mean(test_score))


# In[140]:


get_ipython().run_line_magic('matplotlib', 'inline')
plt.plot(alphas,test_scores)
plt.title('Alphas VS CV Error')


# In[141]:


min(test_scores)


# Using Ridge, the minum score can be around 0.135, while alpha is about 15.

# ## 4.2 Random Forest##

# In[144]:


from sklearn.ensemble import RandomForestRegressor


# In[148]:


max_features = [.1,.2,.3,.4,.5,.6,.7,.8,.9]
test_scores = []
for max_feature in max_features:
    clf = RandomForestRegressor(n_estimators=200, max_features=max_feature)
    test_score = np.sqrt(-cross_val_score(clf, X_train, y_train, cv=5, scoring='neg_mean_squared_error'))
    test_scores.append(np.mean(test_score))


# In[149]:


plt.plot(max_features, test_scores)
plt.title("Max Features vs CV Error");


# In[150]:


min(test_scores)


# Using Random Forest, the minum score can be around 0.137, while max feature is 3.

# ## 4.3 Ensemble##

# We merge the 2 models with their best settings to generate the final model.

# In[154]:


ridge = Ridge(alpha = 15)
rf = RandomForestRegressor(n_estimators = 500, max_features = .3)


# In[157]:


ridge.fit(X_train,y_train)
rf.fit(X_train,y_train)


# Then, we do the final predition, and change back the results to the right distribution. Finnally, we use the 2 presictions' mean as our result.

# In[158]:


y_ridge = np.expm1(ridge.predict(X_test))
y_rf = np.expm1(rf.predict(X_test))


# In[159]:


y_final = (y_ridge + y_rf)/2


# In[164]:


submission_df = pd.DataFrame({'Id':test_df.index,'SalePrice':y_final})
submission_df.head()


# In[165]:


submission_df.shape


# The result looks good. And we export it then submit it.

# In[168]:


submission_df.to_csv('submission.csv',index = False)

