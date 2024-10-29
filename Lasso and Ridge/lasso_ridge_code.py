
import pandas as pd  
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet 
from sklearn.pipeline import Pipeline 
from sklearn.preprocessing import MinMaxScaler 
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer 
from feature_engine.outliers import Winsorizer 
from sklearn.preprocessing import OneHotEncoder 
from sklearn.metrics import r2_score 
import joblib 
import pickle 
import numpy as np 
import matplotlib.pyplot as plt 
import sweetviz
import pyodbc
from sqlalchemy import create_engine,text
from urllib.parse import quote

df=pd.read_csv(r"C:/Users/sukes/OneDrive/Desktop/360/ds Assingments/Lasso and Ridge/50_Startups (1).csv")
user='root'
pw=quote('*********')
db='datascience'
engine=create_engine(f"mysql+pymysql://{user}:{pw}@localhost/{db}")
df.to_sql('startup',con=engine,if_exists='replace',chunksize=1000,index=False)
sql="select * from startup;"
df=pd.read_sql_query(text(sql),engine.connect())
df.isna().sum()
df.columns

lasso_report=sweetviz.analyze(df)
lasso_report.show_html('lasso_report.html')

#Taking inputs from dataframe
X=df.iloc[:,0:4]
X.head()

#Taking ouput from dataframe
Y=df.Profit

numeric_features=X.select_dtypes(exclude=['object']).columns
numeric_features

#Create a pipeline for numerical columns
num_pipeline=Pipeline([('impute', SimpleImputer(strategy='mean')),('scale',MinMaxScaler())])
num_pipeline

#Encoding on State column
categorical_features=['State']
categorical_features

#creating pipeline for categorical columns
categ_pipeline=Pipeline([('encoding',OneHotEncoder(sparse_output=False))])
categ_pipeline

# Using columntransformer to transform the columns of a array or pandas dataframe
preprocess_pipeline=ColumnTransformer([('numerical',num_pipeline,numeric_features),
                                       ('categorical',categ_pipeline,categorical_features)])

# Pass the raw data through pipeline
processed = preprocess_pipeline.fit(X)  

processed
joblib.dump(processed, 'processed_lasso')
import os 
os.getcwd()
df1 = pd.DataFrame(processed.transform(X), columns =  processed.get_feature_names_out())
df1.describe()
df1.iloc[:,0:3].columns

#Visualizing outliers by using of boxplot
plt.boxplot(df1.iloc[:,0:3])

df1.iloc[:,0:3].plot(kind ='box', subplots = True, sharey = False, figsize = (15, 8))
plt.subplots_adjust(wspace = 0.75)
plt.show()

# To retain outliers we use winsorization
winsor = Winsorizer(capping_method = 'iqr', 
                    tail = 'both',
                    fold = 1.5,
                    variables = list(df1.iloc[:,0:3].columns))

# Fit the data
winz_data = winsor.fit(df1[list(df1.iloc[:,0:3].columns)])

#save the pipeline
joblib.dump(winz_data, 'winsor')

df1[list(df1.iloc[:,0:3].columns)] = winz_data.transform(df1[list(df1.iloc[:,0:3].columns)])
df1.iloc[:,0:3].plot(kind ='box', subplots = True, sharey = False, figsize = (15, 8))

#increase spacing between subplots
plt.subplots_adjust(wspace = 0.75)# ws is the width of the padding between subplots,as the fraction of average axis width.
plt.show()
Y.head()

# train_test_split is used to split the data into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(df1, Y, test_size = 0.2, random_state = 0)
X_train.shape
X_test.shape
# Fit a linear regression model
regressor = LinearRegression()
regressor.fit(X_train, Y_train)
# Evaluate the model
# Evaluate the model with train data

pred_train = regressor.predict(X_train)  # Predict on train data

pred_train
# Predict on test set and evaluate performance
y_pred = regressor.predict(X_test)
r2_score_linear = r2_score(Y_test, y_pred)
y_pred
r2_score_linear

# Fit a Lasso regression model with L1 regularization
lasso = Lasso(alpha=0.01)
lasso.fit(X_train, Y_train)

# Predict on test set and evaluate performance
y_pred_lasso = lasso.predict(X_test)
y_pred_lasso 
r2_score_lasso = r2_score(Y_test, y_pred_lasso)
r2_score_lasso

# Fit a Ridge regression model with L2 regularization
ridge = Ridge(alpha=0.01)
ridge.fit(X_train, Y_train)

# Predict on test set and evaluate performance
y_pred_ridge = ridge.predict(X_test)
y_pred_ridge

r2_score_ridge = r2_score(Y_test, y_pred_ridge)
r2_score_ridge

 # Create an instance of the Elastic Net model
enet = ElasticNet(alpha=0.1, l1_ratio=0.5)
elastic_net_model = ElasticNet(alpha=0.1, l1_ratio=0.5)

# Fit the model on the training data
enet.fit(X_train, Y_train)

# Predict on the test data
y_pred_enet = enet.predict(X_test)
y_pred_enet

# Evaluate the model performance
r2_score_elastic_net = r2_score(Y_test, y_pred )
r2_score_elastic_net

# Print R-squared scores for each model
print('Linear Regression R-squared:', r2_score_linear)
print('Lasso Regression R-squared:', r2_score_lasso)
print('Ridge Regression R-squared:', r2_score_ridge)
print('ElasticNet R-squared:', r2_score_elastic_net)

# GridSearchCV is a method in scikit-learn that allows you to search over a grid of hyperparameters for the best combination of parameters for a given model
from sklearn.model_selection import GridSearchCV
# Lasso Regression
# from sklearn.model_selection import GridSearchCV

lasso = Lasso()

parameters = {'alpha': [1e-15, 1e-10, 1e-8, 1e-4, 1e-3, 1e-2, 1, 5 ,10, 20]}
lasso_reg = GridSearchCV(lasso, parameters, scoring = 'r2', cv = 5)
lasso_reg.fit(X_train, Y_train)
lasso_reg.best_params_
lasso_reg.best_score_
y_pred_lasso = lasso_reg.predict(X_test)
# Adjusted r-square#
Grid_lasso = lasso_reg.score(X_train,Y_train)
Grid_lasso
# Ridge Regression
# from sklearn.model_selection import GridSearchCV
# from sklearn.linear_model import Ridge

ridge = Ridge()

parameters = {'alpha': [1e-15, 1e-10, 1e-8, 1e-4, 1e-3, 1e-2, 1, 5 ,10, 20]}
ridge_reg = GridSearchCV(ridge, parameters, scoring = 'r2', cv = 5)
ridge_reg.fit(X_train, Y_train)

ridge_reg.best_params_
ridge_reg.best_score_
ridge_pred = ridge_reg.predict(X_test)

# Adjusted r-square#
Grid_ridge = ridge_reg.score(X_train, Y_train)
Grid_ridge

# ElasticNet Regression
# from sklearn.model_selection import GridSearchCV
# from sklearn.linear_model import ElasticNet
enet = ElasticNet()

# parameters = {'alpha': [1e-15, 1e-10, 1e-8, 1e-4, 1e-3, 1e-2, 1, 5 ,10, 20]}

enet_reg = GridSearchCV(enet, parameters, scoring = 'r2', cv = 5)

enet_reg.fit(X_train, Y_train)
enet_reg.best_params_
enet_reg.best_score_
enet_pred = enet_reg.predict(X_test)

# Adjusted r-square
Grid_elasticnet = enet_reg.score(X_train,Y_train)
Grid_elasticnet
scores_all = pd.DataFrame({'models':['Lasso', 'Ridge', 'Elasticnet', 'Grid_lasso', 'Grid_ridge', 'Grid_elasticnet'], 'Scores':[r2_score_lasso, r2_score_ridge, r2_score_elastic_net, Grid_lasso, Grid_ridge,Grid_elasticnet]})
scores_all
'Best score obtained is for Gridsearch Lasso Regression'
finalgrid = lasso_reg.best_estimator_
finalgrid

# Pickle is a Python module used to convert a Python object into a byte stream representation, which can be saved to a file or transferred over a network
# Save the best model
pickle.dump(finalgrid, open('grid_lasso.pkl', 'wb'))
































