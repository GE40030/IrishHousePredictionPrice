#!/usr/bin/env python
# coding: utf-8

# ## Install the Required Packages

# In[1]:


from sklearn.impute import SimpleImputer

import pandas as pd 
import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt
import sklearn.linear_model
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import OrdinalEncoder
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from datetime import datetime
from sklearn.tree import export_graphviz
import graphviz
from sklearn.tree import plot_tree
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from IPython.display import Image

import plotly.express as px
import plotly.express as px
import plotly
import requests
import json
from sklearn.metrics import explained_variance_score, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor


# In[2]:


get_ipython().system('pip install pydotplus')


# In[3]:


import pydotplus


# In[4]:


get_ipython().system('pip install graphviz')


# In[5]:


get_ipython().system('pip install folium')


# In[6]:


import folium


# In[7]:


get_ipython().system('pip install geopy')


# In[8]:


from geopy.geocoders import Nominatim


# In[9]:


get_ipython().system('which dot')


# ## 

# # 1.Problem Selection

# This dataset is related to the Average House Price Index data from 1975 to 2013,which was published by the (Data.Gov.ie). These data have been sourced by the Irish mortgage lending agencies on loans approved, which were comparing house prices figures from one period to another. 
# The reason, I chose this dataset to be analysed is because, I have developed an interest to find out what are the key factors that has made the Irish house prices increase over the years and which locations have the prices increased the most. I am planning to use multiple data mining michine learning models, such as Linear Regression, Multiple Linear Regression and Decision Tree Regression, which I will be analysing their perfomance and make the right recommendation for the right machine learning model that can be used according its performance.
# After having complete this project and extract useful information, it might suit to any individual or businesses that need to make further analyses regarding the Irish house increase prices market. However, this is just an assumption based on the data available, which we cannot taking as a consideration because there are many other factors, which might exist in the real scenario.   
# 

# # 2. Understanding the nature of the Problem

# On the business understanding phase, is where the people involved have to assess and determine the business objective through a carefully project plan.
# As I had mentioned before, the objective of this assignment is to investigate the performance of the house’s prices over the years and to identify the current situation of the dataset attributes. Therefore, after having complete the objectives, it should contribute to identify the right model which predict the prices of the houses and what are the attributes that have strong relationships that affects the house price increases over the years.
# 

# ## 

# # 3 Data Understanding

# In this phase, the analyst needs to make a deep analysis of the data that has been provided. It is important to understand the dataset and its characteristics, which includes collecting, describing, exploring, and verifying data quality.
# The deliverables of this phase are:
# •	Gathering data
# •	Our data description
# •	data exploration report
# •	Data quality report
# 

# # 

# ## 3.1 Gathering data¶

# In[10]:


df = pd.read_csv("new__houses_average_prices.csv")


# In[11]:


df.head()


# In[12]:


df.shape


# ## 3.2 Describing our data 

# In[13]:


df.info()


# ## 

# ## 3.3 Exploring the data

# ## Q2. PART(A) Data Exploration

# In[14]:


df.describe(include="all")


# In[15]:


df["STATISTIC"].value_counts()


# In[16]:


df["Statistic_Label"].value_counts()


# In[17]:


df["Quarter"].value_counts()


# In[18]:


df["Area"].value_counts()


# ## Finding the area with most highest average house prices 

# In[19]:


##Group the data by the 'Area' column and calculate the mean of the 'Total_Amount' column
area_prices = df.groupby('Area')['Total_Amount'].mean()



# In[20]:


# Find the area with the highest average total amount
most_expensive_area = area_prices.idxmax()


# In[21]:


print("The most expensive place in the country is:", most_expensive_area)


# In[ ]:





# In[ ]:





# In[22]:


fig,ax = plt.subplots(figsize=(15,8))
plt = sns.boxplot(x='Area', y='Total_Amount',linewidth=2.5, data=df);


# In[23]:


sns.set(rc={"figure.figsize":(15, 8)})
sns.boxplot(x='Area',y='Total_Amount', data=df);


# In[24]:


## Find the limits
upper_limit = df['Total_Amount'].mean() + 3*df['Total_Amount'].std()
lower_limit = df['Total_Amount'].mean() - 3*df['Total_Amount'].std()
print('upper limit:', upper_limit)
print('lower limit:', lower_limit)


# In[25]:


median_data = df['Total_Amount'].median()
median_data


# In[26]:


mean_data = df['Total_Amount'].mean()
mean_data


# In[27]:


df['Total_Amount'].min()


# In[165]:


df['Years']-df['Years']/100


# In[ ]:





# # 

# In[28]:


sns.boxplot(x=df['Total_Amount']);


# In[29]:


sns.histplot(df.Years)
sns.despine()


# In[ ]:





# In[30]:


sns.countplot(data=df, y='Statistic_Label')


# In[ ]:





# In[152]:


sns.countplot(data=df, y='Area')


# In[ ]:





# In[129]:


df.skew()


# In[ ]:





# # Plot Histograms.¶

# In[136]:


df['Total_Amount'].mean()


# In[138]:


df['Total_Amount'].median()


# In[134]:


df.hist(column='Total_Amount')

##set labels for x and y axis and the title of the histogram
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.title('Histogram of the Total Amount Price in a Property')

# display the histogram
plt.show()


# The data ranges from 0 to 500,000, with a central tendency between 200,000 and 300,000. The histogram is skewed to the right, indicating that a few values are significantly larger than the majority of values, resulting in a long right tail and a left-shifted peak. In a right-skewed distribution, the mean is usually greater than the median, indicating that extreme values on the right side pull the average towards them. This is common when there is a natural minimum value but no maximum value for a variable, and some observations have much larger values than others. Outliers in the data may cause a right-skewed distribution of Total Price, as shown in the above boxplot.

# In[141]:


df.hist(column='Years')


# In[162]:


sns.histplot(x='Total_Amount', data=df, hue='Area');


# In[ ]:





# # Adding a map with the counties names and areas location 

# In[31]:


# Create a dictionary of counties names and their locations
counties = {
    "Antrim": "54.718545,-6.029635",
    "Armagh": "54.348171,-6.653394",
    "Carlow": "52.717238,-6.836299",
    "Cavan": "53.990839,-7.360419",
    "Clare": "52.807878,-9.280899",
    "Cork": "51.898514,-8.475603",
    "Derry": "55.045175,-7.318268",
    "Donegal": "54.929431,-8.110537",
    "Down": "54.352027,-5.734058",
    "Dublin": "53.349805,-6.260310",
    "Fermanagh": "54.361776,-7.640107",
    "Galway": "53.276533,-9.069362",
    "Kerry": "52.154461,-9.566863",
    "Kildare": "53.158934,-6.909502",
    "Kilkenny": "52.654145,-7.244787",
    "Laois": "53.032355,-7.301690",
    "Leitrim": "54.115869,-8.067642",
    "Limerick": "52.664919,-8.623396",
    "Longford": "53.727036,-7.799250",
    "Louth": "53.877200,-6.414500",
    "Mayo": "53.776449,-9.060580",
    "Meath": "53.605548,-6.656417",
    "Monaghan": "54.248962,-6.968284",
    "Offaly": "53.269034,-7.492790",
    "Roscommon": "53.748831,-8.178189",
    "Sligo": "54.269771,-8.469891",
    "Tipperary": "52.473127,-8.161883",
    "Tyrone": "54.602755,-7.296174",
    "Waterford": "52.259319,-7.110070",
    "Westmeath": "53.534156,-7.344057",
    "Wexford": "52.460070,-6.544670",
    "Wicklow": "52.980820,-6.044590"
}


# In[32]:


# Create a map centered on Ireland
map_ireland = folium.Map(location=[53.4129, -8.2439], zoom_start=7)

# Add markers for each county to the map
geolocator = Nominatim(user_agent="Area")
for county, location in counties.items():
    latitude, longitude = location.split(",")
    location = geolocator.reverse(f"{latitude}, {longitude}")
    folium.Marker(location=[latitude, longitude], popup=county).add_to(map_ireland)

# Show the map
map_ireland


# In[151]:


columns=['Total_Amount']
for i in columns:
    fig = px.treemap(df, values=i, path=['Area'], 
                     title='Treemap for Areas around the country showing property for sale {}'.format(i),
                     width=1000,height=700)
    fig.show()


# Based on the treemap above, the county of Limerick has the highest total amount of houses pricing in the country, which has the total amount of €48,776,870. However, this pricing value was based in 2016, which I believe is much higher now. 

# In[ ]:





# In[34]:


import matplotlib.pyplot as plt
plt.close('all')


# In[35]:


plt.scatter(df['Years'], df['Total_Amount'])
plt.xlabel('Years')
plt.ylabel('Total_Amount')


# ## 

# ## Checking the Correlation between the variables

# In[36]:


df1 = df.corr()


# In[37]:


df1


# In[38]:


sns.heatmap(df1.corr(), cmap='Blues', annot=True);


# In[ ]:





# # 

# ## 3.4 Data Quality¶

# # 

# ### Checking Nan values

# In[39]:


df.isna().sum()


# As we can see from our dataset, there are 322 missing values from the C02343V02817 and 1 missing values regarding to Total Amount column, which needs to be handled accordingly. 

# In[40]:


df.isna().any()


# In[41]:


df1 = pd.DataFrame(df)


# In[42]:


##Function to find out the percentage of null values in the dataframe

NullValues=df1.isnull().sum()/len(df1)


# In[43]:


NullValues


# ### Checking Duplicates values

# In[44]:


df1.duplicated().any()


# In[45]:


df1.duplicated().sum()


# # 

# # 4 Preparing data for machine learning

# ## Data selection:

# In[46]:


##Creating a new DataFrame as I will be dropping the STATISTIC column. Because it is not useful for the purpose of our machine learning
df2 =df1.drop(['STATISTIC'], axis=1)
df2


# In[47]:


## Renaming the Statistic Label and C02343V02817 columns to make it suitable for the analyses later on 
df2.rename(columns={'Statistic_Label': 'Description', 'C02343V02817': 'Codes'}, inplace=True)


# In[48]:


df2


# In[49]:


df2.info()


# In[50]:


df2.dtypes


# In[ ]:





# ## data Cleaning

# In[51]:


df2.isna().sum()


# In[52]:


df2.dropna(subset=['Codes'])

df2


# In[53]:


df2['Total_Amount'] = df2['Total_Amount'].interpolate()


# In[54]:


df2.isna().sum()


# In[55]:


df2.dropna(subset=['Codes'], inplace=True)

df2.head()


#  For the missing values treatments, for example, I decided to remove the rows with missing values within the Codes column and used the interpolate function within the Total Amount column, which estimates the unknown value in the same increasing order from previous values.

# In[56]:


df2.isna().any()


# In[57]:


df2.isnull().sum()


# In[58]:


df2.duplicated().any()


# In[59]:


df2.duplicated().sum()


# In[60]:


df2.head()


# In[61]:


df2.shape


# ## 

# ## Encoding the dataset

# In[62]:


value_counts = df2['Total_Amount'].value_counts()
value_counts


# In[63]:


##Creating a new DataFrame from a our dataset, which will be encoded to make it suitable for the machine learning


# In[64]:


df3 = pd.DataFrame(df2)


# In[65]:


# One-hot encode the Description, Quarter, and Area columns
one_hot = pd.get_dummies(df3, columns=["Description", "Quarter", "Area"], drop_first=True)


# In[66]:


##Creating a new DataFrame from a our dataset, which will be encoded to make it suitable for the machine learning
df4 = pd.DataFrame(one_hot)


# In[67]:


df4.head(10)


# In[68]:


df4.shape


# In[69]:


df4.columns


# In[70]:


df5 = df4.corr()


# In[71]:


df5


# In[72]:


sns.heatmap(df5.corr(), cmap='Blues', annot=True);


# In[73]:


sns.regplot(x='Codes', y='Total_Amount', data=df5)


# Based on the analysis carried out, it is clear that the Codes column has weak correlation with total amount column. This suggest that if the total amount increases, the code column will not be affected.

# In[ ]:





# In[ ]:





# ## Correlation Matrix 

# In[74]:




df5['Total_Amount'].sort_values(ascending=False)


# I will be using the Correlation Matrix to select the best column to be used in the machine learning models

# # 

# # Q5. Preparing and parameter selection for the models

# # Model Feature selection:¶

# For the Simple Linear Regression modelling, It has been decided to use the independent variable "Years" column because it has a strong correlation with the dependent variable "Total_Amount" column 

# In[75]:


features = ['Years']


# In[ ]:





# In[76]:


##Droping the Total Amount column to be the target value in the machine learning model
## Creating an Y axis from the Total_Amount as it will be required later to be analysied in the Regression analysis

houseprice_scaled= pd.DataFrame(df4['Total_Amount'])

houseprice_scaled


# In[77]:


# Separating out the features
x_data = df4[features].values

x_data


# In[78]:


# Separating out the target
y_data = houseprice_scaled['Total_Amount'].values
y_data 


# # 

# # 

# # Model Paramenter Creation

# In[79]:


X_train, X_test, y_train, y_test = train_test_split(x_data,y_data, test_size=0.3,random_state=0)
print('X_train:', X_train)
print('y_train:', y_train)


# In[80]:


print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)


# In[ ]:





# # 

# ### Normalize and scale the dataset

# In[81]:


#Standard Scale the data for Linear Regression Model
# Create an object of StandardScaler class
scaler=StandardScaler()

#means apply standard scaler for X_train and X_test data
X_train_scaled=scaler.fit_transform(X_train)
X_test_scaled=scaler.transform(X_test)


# In[ ]:





# # Model building¶

# ### Simple Linear Regression

# In[82]:




# Create an instance of the LinearRegression class
simple_linear_model = LinearRegression()

# Fit the model to the scaled training data
simple_linear_model.fit(X_train_scaled, y_train)

# Use the model to make predictions on the scaled test data
simple_prediction_model = simple_linear_model.predict(X_test_scaled)


# In[83]:


simple_prediction_model


# # 

# # Simple Linear Regression Evaluation Metrics

# In[84]:


slr_mse = mean_squared_error(y_test, simple_prediction_model)
slr_r2 = r2_score(y_test, simple_prediction_model)
slr_evs = explained_variance_score(y_test, simple_prediction_model)
slr_mae = mean_absolute_error(y_test, simple_prediction_model)
slr_rmse = np.sqrt(slr_mse)
slr_mape = np.mean(np.abs((y_test - simple_prediction_model) / y_test)) * 100

print("Simple Linear Regression MSE: {:.4f}".format(slr_mse))
print("Simple Linear Regression R-squared:", slr_r2)
print("Simple Linear Regression Explained Variance Score:", slr_evs)
print("Simple Linear Regression Mean Absolute Error:", slr_mae)
print(f"Simple Linear Regression Root Mean Squared Error: {slr_rmse}")
print(f"Simple Linear Regression Mean Absolute Percentage Error: {slr_mape}%")
# Calculate and display accuracy
accuracy=0
accuracy = 100 - np.mean(slr_mape)
print('Accuracy:', round(accuracy, 2), '%.')


# # 

# #### Apply Cross Validation to the Simple Linear Regression Model

# In[85]:


from sklearn.model_selection import KFold, cross_val_score
k_folds = KFold(n_splits = 10)

scores = cross_val_score(simple_linear_model, X_train_scaled, y_train, cv = k_folds)

print("Cross Validation Scores: ", scores)
print("Average CV Score: ", scores.mean())
print("Standard Deviation CV Score: ", scores.std())
print("Number of CV Scores used in Average: ", len(scores))


# In[ ]:





# # 

# ## Visualise the Simple Linear Regression results

# In[86]:


import matplotlib.pyplot as plt
plt.close('all')


# In[87]:


plt.figure(figsize=(15,10))
plt.scatter(y_test, simple_prediction_model)
slope, intercept = np.polyfit(y_test, simple_prediction_model, 1)
plt.plot(simple_prediction_model, slope*simple_prediction_model + intercept, color='red')
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.title('Actual vs Predicted')
plt.show()


# In[88]:


# Plot the predicted values against the true values for the test set
plt.scatter(y_test, simple_prediction_model)

# Add a trend line
p = np.polyfit(y_test, simple_prediction_model, 1)  # Fit a first-degree polynomial to the data
plt.plot(y_test, np.polyval(p, y_test), 'r')  # Plot the line

# Add a diagonal line for comparison
plt.plot([0, 50], [0, 50], '--', color='gray')

plt.xlabel('True values')
plt.ylabel('Predicted values')
# format the tick labels to display the full values
plt.ticklabel_format(useOffset=False)

plt.show()


# In[ ]:





# In[89]:


# set the float format to display 10 decimal places
pd.options.display.float_format = '{:.10f}'.format
pred_df=pd.DataFrame({'Actual Value':y_test,'Predicted Value':simple_prediction_model,'Difference':y_test-simple_prediction_model})
pred_df


# In[ ]:





# # 

# # Multiple Linear Regression

# For the multiple linear regression model, I will be using all the columns for our features variables because they have strong correlation with the target column.

# In[90]:


features1 = ['Years', 'Codes',
       'Description_Second Hand House Prices', 'Quarter_Q2', 'Quarter_Q3',
       'Quarter_Q4', 'Area_Dublin', 'Area_Galway', 'Area_Limerick',
       'Area_Other areas', 'Area_Waterford']


# In[91]:


x_data1 = df4[features1].values
x_data1


# In[92]:


# Separating out the target
y_data1 = houseprice_scaled['Total_Amount'].values
y_data1 


# # 

# ## Split dataset into training and testing sets¶

# In[93]:


X_train, X_test, y_train, y_test = train_test_split(x_data1,y_data1, test_size=0.3,random_state=0)


# In[94]:


print('X_train:', X_train)
print('y_train:', y_train)


# In[95]:


print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)


# # 

# ## Normalize and scale the dataset¶

# In[96]:


#Standard Scale the data for Linear Regression Model
# Create an object of StandardScaler class
scaler=StandardScaler()

#means apply standard scaler for X_train and X_test data
X_train_scaled1=scaler.fit_transform(X_train)
X_test_scaled1=scaler.transform(X_test)


# In[ ]:





# # Model Building

# ### Multiple Linear Regression

# In[97]:


# Create an instance of the LinearRegression class
Multiple_linear_model = LinearRegression()

# Fit the model to the scaled training data
Multiple_linear_model.fit(X_train_scaled1, y_train)

# Use the model to make predictions on the scaled test data
Multiple_prediction_model = Multiple_linear_model.predict(X_test_scaled1)


# In[98]:


Multiple_prediction_model


# # 

# # Multiple Linear Regression Evaluation Metrics

# In[99]:


mlr_mse = mean_squared_error(y_test, Multiple_prediction_model)
mlr_r2 = r2_score(y_test, Multiple_prediction_model)
mlr_evs = explained_variance_score(y_test, Multiple_prediction_model)
mlr_mae = mean_absolute_error(y_test, Multiple_prediction_model)
mlr_rmse = np.sqrt(mlr_mse)
mlr_mape = np.mean(np.abs((y_test - Multiple_prediction_model) / y_test)) * 100

print("Multiple Linear Regression MSE: {:.4f}".format(mlr_mse))
print("Multiple Linear Regression R-squared:", mlr_r2)
print("Multiple Linear Regression Explained Variance Score:", mlr_evs)
print("Multiple Linear Regression Mean Absolute Error:", mlr_mae)
print(f"Multiple Linear Regression Root Mean Squared Error: {mlr_rmse}")
print(f"Multiple Linear Regression Mean Absolute Percentage Error: {mlr_mape}%")
# Calculate and display accuracy
accuracy=0
accuracy = 100 - np.mean(mlr_mape)
print('Accuracy:', round(accuracy, 2), '%.')


# # 

# #### Apply Cross Validation to the Multiple Linear Regression Model

# In[100]:


from sklearn.model_selection import KFold, cross_val_score
k_folds = KFold(n_splits = 10)

scores = cross_val_score(Multiple_linear_model, X_train_scaled1, y_train, cv = k_folds)

print("Cross Validation Scores: ", scores)
print("Average CV Score: ", scores.mean())
print("Standard Deviation CV Score: ", scores.std())
print("Number of CV Scores used in Average: ", len(scores))


# # 

# ## Visualise the Multiple Linear regression results

# In[101]:


plt.figure(figsize=(15,10))
plt.scatter(y_test, Multiple_prediction_model)
slope, intercept = np.polyfit(y_test, Multiple_prediction_model, 1)
plt.plot(Multiple_prediction_model, slope*Multiple_prediction_model + intercept, color='red')
plt.xlabel('Actual')
plt.ylabel('Predictied')
plt.title('Actual vs Predicted')

plt.show()


# # 

# In[102]:


# set the float format to display 10 decimal places
pd.options.display.float_format = '{:.10f}'.format
multip_df=pd.DataFrame({'Actual Value':y_test,'Predicted Value':Multiple_prediction_model,'Difference':y_test-Multiple_prediction_model})
multip_df


# # 

# # 

# # Decision Tree Regression Model

# For the decision tree regression model, it has been decided to use all the columns for our features variables because they included the negative and strong correlated columns that can be used for our machine learning build process.

# In[103]:


# Create an instance of the DecisionTreeRegressor class
decision_tree_model = DecisionTreeRegressor(max_depth=3, random_state=42)


# In[104]:


# Fit the model to the training data
decision_tree_model.fit(X_train,y_train)


# In[105]:


# Use the model to make predictions on the test data
decision_tree_prediction_model = decision_tree_model.predict(X_test)


# In[106]:


decision_tree_prediction_model


# In[ ]:





# # Decision Tree Evaluation Metrics

# In[107]:



dt_mse = mean_squared_error(y_test, decision_tree_prediction_model)
dt_r2 = r2_score(y_test, decision_tree_prediction_model)
dt_evs = explained_variance_score(y_test, decision_tree_prediction_model)
dt_mae = mean_absolute_error(y_test, decision_tree_prediction_model)
dt_rmse = np.sqrt(dt_mse)
dt_mape = np.mean(np.abs((y_test - decision_tree_prediction_model) / y_test)) * 100

print("Decision Tree Regression MSE:", dt_mse)
print("Decision Tree Regression R-squared:", dt_r2)
print("Decision Tree Regression Explained Variance Score:", dt_evs)
print("Decision Tree Regression Mean Absolute Error:", dt_mae)
print(f"Decision Tree Regression Root Mean Squared Error: {dt_rmse}")
print(f"Decision Tree Regression Mean Absolute Percentage Error: {dt_mape}%")
# Calculate and display accuracy
accuracy=0
accuracy = 100 - np.mean(dt_mape)
print('Accuracy:', round(accuracy, 2), '%.')


# # 

# #### Apply Cross Validation to the Decision Tree Regression Model

# In[108]:


k_folds = KFold(n_splits = 10)
scores = 0
scores = cross_val_score(decision_tree_model, X_train, y_train, cv = k_folds)

print("Cross Validation Scores: ", scores)
print("Average CV Score: ", scores.mean())
print("Standard Deviation CV Score: ", scores.std())
print("Number of CV Scores used in Average: ", len(scores))


# In[ ]:





# In[109]:


# set the float format to display 10 decimal places
pd.options.display.float_format = '{:.10f}'.format
dt_df=pd.DataFrame({'Actual Value':y_test,'Predicted Value':decision_tree_prediction_model,'Difference':y_test-decision_tree_prediction_model})
dt_df


# ## Visualise the decision tree regression results

# In[110]:


plt.figure(figsize=(20,15))
plot_tree(decision_tree_model, filled=True, feature_names=features1)
plt.show()


# In[ ]:





# In[111]:



from sklearn.tree import export_graphviz


dot_data = export_graphviz(decision_tree_model, out_file=None, filled=True, rounded=True, special_characters=True)
graph = pydotplus.graph_from_dot_data(dot_data)

graph.set_size('"60,10!"')
Image(graph.create_png())


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[112]:


plt.figure(figsize=(15,10))
plt.scatter(y_test, decision_tree_prediction_model)
slope, intercept = np.polyfit(y_test, decision_tree_prediction_model, 1)
plt.plot(decision_tree_prediction_model, slope*decision_tree_prediction_model + intercept, color='red')
plt.xlabel('Actual')
plt.ylabel('Predictied')
plt.title('Actual vs Predicted')

plt.show()


# The Decision Tree regresssion has shown a positive strong correlation, which is 0.9996958663408702. However, the Multiple Linear Regression has the R-squared of 1.0, which represents the best fit correlation acompared to the Decision Tree Regression model.
# 

# In[ ]:





# # Build Gradient Boosting Regression Model

# In[113]:


## Gradient Boosting Regression
gb_model = GradientBoostingRegressor()
gb_model.fit(X_train, y_train)
gb_prediction_model = gb_model.predict(X_test)



# In[114]:


gb_prediction_model 


# In[115]:


gb_mse = mean_squared_error(y_test, gb_prediction_model)
gb_r2 = r2_score(y_test, gb_prediction_model)
gb_evs = explained_variance_score(y_test, gb_prediction_model)
gb_mae = mean_absolute_error(y_test, gb_prediction_model)
gb_rmse = np.sqrt(gb_mse)
gb_mape = np.mean(np.abs((y_test - gb_prediction_model) / y_test)) * 100
print("Gradient Boosting Regression MSE:", gb_mse)
print("Gradient Boosting Regression R-squared:", gb_r2)
print("Gradient Boosting Regression Explained Variance Score:", gb_evs)
print("Gradient Boosting Regression Mean Absolute Error:", gb_mae)
print(f"Gradient Boosting Regression Root Mean Squared Error: {gb_rmse}")
print(f"Gradient Boosting Regression Mean Absolute Percentage Error: {gb_mape}%")
# Calculate and display accuracy
accuracy=0
accuracy = 100 - np.mean(gb_mape)
print('Accuracy:', round(accuracy, 2), '%.')


# #### Apply Cross Validation to the Gradient Boosting Regression Model

# In[116]:


k_folds = KFold(n_splits = 10)
scores = 0
scores = cross_val_score(gb_model, X_train, y_train, cv = k_folds)

print("Cross Validation Scores: ", scores)
print("Average CV Score: ", scores.mean())
print("Standard Deviation CV Score: ", scores.std())
print("Number of CV Scores used in Average: ", len(scores))


# In[ ]:





# In[117]:


# Plot the predicted vs actual values for Gradient Boosting Regression Model
plt.scatter(y_test, gb_prediction_model)
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.title("Gradient Boosting Regression")
# Add a diagonal line with the same x and y values
min_val = min(np.min(y_test), np.min(gb_prediction_model))
max_val = max(np.max(y_test), np.max(gb_prediction_model))
plt.plot([min_val, max_val], [min_val, max_val], 'k--')

plt.show()


# In[118]:


plt.figure(figsize=(15,10))
plt.scatter(y_test, gb_prediction_model)
slope, intercept = np.polyfit(y_test, gb_prediction_model, 1)
plt.plot(gb_prediction_model, slope*gb_prediction_model + intercept, color='red')
plt.xlabel('Actual')
plt.ylabel('Predictied')
plt.title('Actual vs Predicted')

plt.show()


# In[ ]:





# In[ ]:





# ## Model evaluation using performance metrics

# In[ ]:





# In[127]:


# Create a dataframe to display the results
from tabulate import tabulate
results = 0
results = pd.DataFrame({
    'Model': ['Linear Regression', 'Multiple Linear Regression', 'Decision Tree Regression', 'Gradient Boosting Regression Model'],
    'MSE':       [slr_mse, mlr_mse, dt_mse, gb_mse],
    'R-squared': [slr_r2, mlr_r2, dt_r2, gb_r2],
    'EVS':       [slr_evs, mlr_evs, dt_evs, gb_evs],
    'MAE':       [slr_mae, mlr_mae, dt_mae, gb_mae],
    'RMSE':      [slr_rmse, mlr_rmse, dt_rmse, gb_rmse],
    'MAPE':      [slr_mape, mlr_mape, dt_mape, gb_mape]
})

print(tabulate(results, headers='keys', tablefmt='psql'))


# Based on the information presented in the table above, it appears that the four models exhibit varying levels of performance and accuracy when it comes to predicting a certain variable (which has been explained earlier in the cross-validation models).
# The Linear Regression model (Model 0) has a Mean Squared Error (MSE) of 3.26, a Root Mean Squared Error (RMSE) of 57,117.3, and an accuracy of 54.23%. These values suggest that this model has one of the highest levels of error in its predictions based on the table information. However, it still has a relatively high R-squared value of 0.70, a lower Explained Variance Score (EVS) value of 0.70, a Mean Absolute Error (MAE) of 40,272.2, and a Mean Absolute Percentage Error (MAPE) value of 45.77. These values indicate that it explains a significant proportion of the variance in the dependent variable.
# The Multiple Linear Regression model (Model 1) has a lower MSE of 3.04 and a lower RMSE of 55,177.3 compared to the Linear Regression model, indicating that it has a higher level of prediction accuracy of 54.23%. Additionally, it has a higher R-squared value of 0.72, a higher EVS of 0.72, and a higher MAPE of 50.71. However, the MAE value is the same compared to the Linear Regression model, which suggests that it has a better ability to explain the variation in the dependent variable.
# The Decision Tree Regression model (Model 2) has even lower values of MSE of 1.81, an RMSE value of 42,574.8, and an accuracy of 72.73% compared to both Linear Regression and Multiple Linear Regression models. Furthermore, it has the highest R-squared value of 0.83, an EVS value of 0.83, and the lowest MAPE value of 27.27, among the other two models mentioned above. These values indicate that it explains the variance in the dependent variable more effectively.
# Finally, the Gradient Boosting Regression Model (Model 3) has the highest values of MSE of 6.59, an accuracy of 87.13%, and the lowest RMSE of 25,676. These values suggest that it has the most accurate predictions among all models. Additionally, it has a very high R-squared value of 0.94 and an EVS value of 0.93. However, it has the lowest MAE value of 13,444.9 and an MAPE value of 12.8705, indicating that it explains a large proportion of the variance in the dependent variable.
# Overall, based on the table information, it appears that the Gradient Boosting Regression Model (Model 3) has the best performance in terms of prediction accuracy and ability to explain the variation in the dependent variable. However, it is important to note that this conclusion may change if we consider other metrics or if we assess the models' performance on a different dataset.
# 
