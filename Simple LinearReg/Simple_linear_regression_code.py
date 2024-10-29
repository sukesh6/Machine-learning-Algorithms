
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sweetviz as sv
from feature_engine.outliers import Winsorizer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
import sklearn.metrics as skmet
import joblib
import pickle
from sklearn.model_selection import GridSearchCV
from sqlalchemy import create_engine
from urllib.parse import quote

# IMPORTING DATASET
data = pd.read_csv(r"delevery_time.csv")

from sqlalchemy import create_engine
user = 'root'
pw = '*********'
db = 'datascience'

engine = create_engine(f"mysql+pymysql://{user}:%s@localhost/{db}" % quote(f'{pw}'))

data.to_sql('Delivery_Time'.lower(), con = engine, if_exists = 'replace', index = False)

sql = 'select * from Delivery_Time'.lower()
data = pd.read_sql_query(sql, con = engine)

data.info() # No null values
data.describe()
data.columns

data.rename(columns = {'Delivery Time':'Delivery_Time'}, inplace = True)

data.rename(columns = {'Sorting Time':'Sorting_Time'}, inplace = True)

#Graphical Representation
import matplotlib.pyplot as plt 

plt.bar(height = data['Delivery_Time'], x = np.arange(0, 21, 1))

plt.hist(data['Delivery_Time'],color='skyblue',edgecolor='black')

plt.boxplot(data['Delivery_Time']) #boxplot

plt.bar(height = data['Sorting_Time'], x = np.arange(0, 21, 1))
plt.hist(data['Sorting_Time'],color='green',edgecolor='black') #histogram
plt.boxplot(data['Sorting_Time']) #boxplot

# Scatter plot
plt.scatter(x = data['Sorting_Time'], y = data['Delivery_Time'], color = 'green') 

# correlation
np.corrcoef(data['Sorting_Time'],data['Delivery_Time']) 

# Covariance
# NumPy does not have a function to calculate the covariance between two variables directly. 
# Function for calculating a covariance matrix called cov() 
# By default, the cov() function will calculate the unbiased or sample covariance between the provided random variables.

cov_output = np.cov(data['Sorting_Time'], data['Delivery_Time'])[0, 1]
cov_output

# AUTO EDA
eda = sv.analyze(data)
eda.show_html()

# CHECKING FOR DUPLICATES
duplicate = data.duplicated()
duplicate
sum(duplicate) # Found 0 duplicate


# x = Sorting_Time
# y = Delivery_Time
# SEPERATING INPUTS AND OUTPUT
data.info()
X = data[['Sorting_Time']] # Predictors
Y = data[['Delivery_Time']] # Target



###############################################################################


numeric_features = X.select_dtypes(exclude = ['object']).columns
numeric_features

num_pipeline = Pipeline(steps = [('impute', SimpleImputer(strategy = 'mean'))])
preprocessor = ColumnTransformer(transformers = [('num', num_pipeline, numeric_features)])
processed_slr = preprocessor.fit(X)

joblib.dump(processed_slr, 'processed_slr')

data['Sorting_Time'] = processed_slr.transform(X)


####BUILDING PIPELINE FOR OUTLIERS

# Multiple boxplots in a single visualization.
# Columns with larger scales affect other columns. 
# Below code ensures each column gets its own y-axis.

# pandas plot() function with parameters kind = 'box' and subplots = True
plt.boxplot(data)
data.plot(kind = 'box', subplots = True, sharey = False, figsize = (10, 6)) 
plt.subplots_adjust(wspace = 0.75)
plt.show()

winsor = Winsorizer(capping_method = 'iqr', 
                          tail = 'both',
                          fold = 1.5,
                          variables = ['Sorting_Time'])

winsor = winsor.fit(data[['Sorting_Time']])

joblib.dump(winsor, 'winsor')

data[['Sorting_Time']] = winsor.transform(data[['Sorting_Time']])





# Import library
import statsmodels.formula.api as smf

# Simple Linear Regression
model = smf.ols('Delivery_Time ~ Sorting_Time', data = data).fit()
model.summary()

pred1 = model.predict(pd.DataFrame(data['Sorting_Time']))

# Regression Line
plt.scatter(data.Sorting_Time,data.Delivery_Time)
plt.plot(data.Sorting_Time, pred1, "r")
plt.legend(['Predicted line', 'Observed data'])
plt.show()

# Error calculation
res1 = data.Delivery_Time - pred1
res_sqr1 = res1 * res1
mse1 = np.mean(res_sqr1)
rmse1 = np.sqrt(mse1)
rmse1

######### Model building on Transformed Data
# Log Transformation
# x = log(Sorting_Time); y = Delivery_Time

plt.scatter(x = np.log(data['Sorting_Time']), y = data['Delivery_Time'], color = 'brown')
np.corrcoef(np.log(data.Sorting_Time), data.Delivery_Time) #correlation

model2 = smf.ols('Delivery_Time ~ np.log(Sorting_Time)', data = data).fit()
model2.summary()

pred2 = model2.predict(pd.DataFrame(data['Sorting_Time']))

# Regression Line
plt.scatter(np.log(data.Sorting_Time), data.Delivery_Time)
plt.plot(np.log(data.Sorting_Time), pred2, "r")
plt.legend(['Predicted line', 'Observed data'])
plt.show()

# Error calculation
res2 = data.Delivery_Time - pred2
res_sqr2 = res2 * res2
mse2 = np.mean(res_sqr2)
rmse2 = np.sqrt(mse2)
rmse2


#### Exponential transformation
# x = Sorting_Time; y = log(Delivery_Time)

plt.scatter(x = data['Sorting_Time'], y = np.log(data['Delivery_Time']), color = 'orange')
np.corrcoef(data.Sorting_Time, np.log(data.Delivery_Time)) #correlation

model3 = smf.ols('np.log(Delivery_Time) ~ Sorting_Time', data = data).fit()
model3.summary()

pred3 = model3.predict(pd.DataFrame(data['Sorting_Time']))
pred3_at = np.exp(pred3)
pred3_at

# Regression Line
plt.scatter(data.Sorting_Time, np.log(data.Delivery_Time))
plt.plot(data.Sorting_Time, pred3, "r")
plt.legend(['Predicted line', 'Observed data'])
plt.show()

# Error calculation
res3 = data.Delivery_Time - pred3_at
res_sqr3 = res3 * res3
mse3 = np.mean(res_sqr3)
rmse3 = np.sqrt(mse3)
rmse3


#### Polynomial transformation
# x = Sorting_Time; x^2 = Sorting_Time*Sorting_Time; y = log(Delivery_Time)

model4 = smf.ols('np.log(Delivery_Time) ~ Sorting_Time + I(Sorting_Time*Sorting_Time)', data = data).fit()
model4.summary()

pred4 = model4.predict(pd.DataFrame(data))
pred4_at = np.exp(pred4)
pred4_at

# # Regression line
# from sklearn.preprocessing import PolynomialFeatures
# poly_reg = PolynomialFeatures(degree = 2)
# X = data.iloc[:, 0:1].values
# X_poly = poly_reg.fit_transform(X)
# # y = data.iloc[:, 1].values


plt.scatter(data.Sorting_Time, np.log(data.Delivery_Time))
plt.plot(X, pred4_at, color = 'red')
plt.legend(['Predicted line', 'Observed data'])
plt.show()


# Error calculation
res4 = data.Delivery_Time - pred4_at
res_sqr4 = res4 * res4
mse4 = np.mean(res_sqr4)
rmse4 = np.sqrt(mse4)
rmse4


# Choose the best model using RMSE
data1 = {"MODEL":pd.Series(["SLR", "Log model", "Exp model", "Poly model"]), "RMSE":pd.Series([rmse1, rmse2, rmse3, rmse4])}
table_rmse = pd.DataFrame(data1)
table_rmse

###################
# The best model

from sklearn.model_selection import train_test_split

train, test = train_test_split(data, test_size = 0.2)

finalmodel = smf.ols('Delivery_Time ~ np.log(Sorting_Time)', data = data).fit()
finalmodel.summary()

# Predict on test data
test_pred = finalmodel.predict(pd.DataFrame(test))
pred_test_AT = np.exp(test_pred)
pred_test_AT

# Model Evaluation on Test data
test_res = test.Delivery_Time - test_pred
test_sqrs = test_res * test_res
test_mse = np.mean(test_sqrs)
test_rmse = np.sqrt(test_mse)
test_rmse


# Prediction on train data
train_pred = finalmodel.predict(pd.DataFrame(train))
pred_train_AT = np.exp(train_pred)
pred_train_AT

# Model Evaluation on train data
train_res = train.Delivery_Time - train_pred
train_sqrs = train_res * train_res
train_mse = np.mean(train_sqrs)
train_rmse = np.sqrt(train_mse)
train_rmse


finalmodel.save("model.pickle")
# to load model
from statsmodels.regression.linear_model import OLSResults
model = OLSResults.load("model.pickle")

import os
os.getcwd()


