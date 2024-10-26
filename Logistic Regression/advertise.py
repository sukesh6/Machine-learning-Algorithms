
import pandas as pd
import numpy as np
import sweetviz
import matplotlib.pyplot as plt
from feature_engine.outliers import Winsorizer
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
import joblib
import seaborn as sns
import pandas as pd

# import statsmodels.formula.api as smf
import statsmodels.api as sm
from sklearn.model_selection import train_test_split # train and test

#import pylab as pl
from sklearn import metrics
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.metrics import roc_curve
from sklearn.metrics import classification_report

# SQL Integration
from sqlalchemy import create_engine
from urllib.parse import quote 
from getpass import getpass

advertise=pd.read_csv(r'C:/Users/sukes/OneDrive/Desktop/360/ds Assingments/Logistic Regression/key/logistic regression/advertising.csv')
user_name = 'root'
database = 'datascience'
your_password =quote('************')
engine = create_engine(f'mysql+pymysql://{user_name}:{your_password}@localhost/{database}')
advertise.to_sql('advertise_tbl',con=engine,if_exists='replace',chunksize=1000,index=False)
# Load the offline data into Database 
# adver = pd.read_csv('advertising.csv')
# # adver.info()

# adver.to_sql('advertising', con = engine, if_exists = 'replace', chunksize = 1000, index = False)

### Read the Table (data) from MySQL database
sql = 'SELECT * FROM advertising'

# Convert columns to best possible dtypes using
advertise = pd.read_sql_query(sql, engine)

advertise.head()

advertise.info()
advertise.describe()


# AutoEDA
# Automated Libraries
# import sweetviz
my_report = sweetviz.analyze([advertise, "advertise"])
my_report.show_html('Report.html')


advertise.isna().sum()

# advertise['Ad_Topic_Line'].value_counts()

# advertise['City'].value_counts()

advertise['Country'].value_counts()


#lets remove the not required variables
advertise = advertise.drop(['Ad_Topic_Line', 'City','Timestamp','Country'], axis=1)
advertise['Age'] = pd.to_numeric(advertise['Age'], downcast='float')


#Predictors
 
X = advertise.iloc[:,0:5]

#  Target
Y = advertise[['Clicked_on_Ad']]

X.info()
X.dtypes
#Segregating
categorical_features = X.select_dtypes(include = ['object']).columns

categorical_features

#Seperating Integer and Float data 
numeric_features1 = X.select_dtypes(include = ['float']).columns
numeric_features1

numeric_features2 = X.select_dtypes(include = ['int64']).columns
numeric_features2

# Imputation techniques to handle missing data
# Mode imputation for Integer (categorical) data
# Mean imputation for continuous (Float, data)

num_pipeline1 = Pipeline(steps=[('impute1' , SimpleImputer(strategy = 'mean')), ('scale', MinMaxScaler())])

# Median imputation for Integer (categorical) data
num_pipeline2 = Pipeline(steps=[('impute2' , SimpleImputer(strategy = 'median')), ('scale', MinMaxScaler())])

# # Mode imputation for 'categorical' data
cat_pipeline = Pipeline(steps = [('impute3' , SimpleImputer(strategy = 'most_frequent')),('encoding', OneHotEncoder(sparse_output = False))])


# 1st Imputations Transformer

preprocessor = ColumnTransformer([
    ('num1', num_pipeline1, numeric_features1),
    ('num2', num_pipeline2, numeric_features2),('categorical', cat_pipeline, categorical_features)])

print(preprocessor)

#Fit the data to train imputation pipeline mode1
imp_enc_scale = preprocessor.fit(X)

#Save the pipeline
joblib.dump(imp_enc_scale,'imp_enc_scale')

# Transform the original data
X1 = pd.DataFrame(imp_enc_scale.transform(X), columns = imp_enc_scale.get_feature_names_out())

X1.iloc[:,0:4]

# Multiple boxplots in a single visualization.
#Columns with larger scales affevt other columns.
#Below code ensures each column gets its own y-axis.
# pandas plot() function with parameters kind ='box' and subplots = True

X1.iloc[:,0:4].plot(kind ='box', subplots = True, sharey = False, figsize = (15, 8))

''' sharey True or 'all': x- or y-axis will be shared among all subplots.
False or 'none': each subplot x- or y-axis will be independent.'''

#increase spacing between subplots
plt.subplots_adjust(wspace = 0.75)# ws is the width of the padding between subplots,as the fraction of average axis width.
plt.show()

winsor = Winsorizer(capping_method = 'iqr', # choose IQR rule boundaries or
                    tail = 'both', # cap left,right or both tails
                    fold = 1.5,
                    variables = list(X1.iloc[:,0:4].columns))

# Fit the data
winz_data = winsor.fit(X1[list(X1.iloc[:,0:4].columns)])

#save the pipeline
joblib.dump(winz_data, 'winsor')

X1[list(X1.iloc[:,0:4].columns)] = winz_data.transform(X1[list(X1.iloc[:,0:4].columns)])
X1.info()

# Boxplot
X1.iloc[:,0:4].plot(kind = 'box', subplots = True, sharey = False, figsize = (15, 8))
plt.subplots_adjust(wspace = 0.75) # ws is the width of the padding between subplots,

plt.show()

## Statsmodel
# Building the model and fitting the data

logit_model = sm.Logit(Y, X1).fit()

# Save the model
logit_model.save('logit_model.pkl')


# Summary

logit_model.summary()
logit_model.summary2() # for AIC


# Prediction

pred = logit_model.predict(X1)
pred # Probabilities

# ROC Curve to identify the appropriate cutoff value
fpr , tpr, thresholds = roc_curve(Y, pred)
optimal_idx = np.argmax(tpr-fpr)
optimal_threshold = thresholds[optimal_idx]
optimal_threshold
######## optimal_threshold = 0.41346512057627866


auc = metrics.auc(fpr, tpr)
print("Area under the ROC Curve : %f" % auc)
 # Filling all the cells with zeroes
X1["pred"] = np.zeros(len(X1))  

#taking threshold value and above the prob value will be
X1.loc[pred>optimal_threshold, "pred"] = 1

#confusion matrix
confusion_matrix(X1.pred, Y)
#accuracy score of the model
print('Test accuracy = ',accuracy_score(X1.pred, Y))
#classification report
classification = classification_report(X1["pred"], Y)
print(classification)

### PLOT FOR ROC
plt.plot(fpr, tpr, label = "AUC="+str(auc))
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.legend(loc = 4)
plt.show()

# Data split
x_train, x_test, y_train, y_test = train_test_split (X1, Y,
                                                     test_size = 0.2,
                                                     random_state = 0,
                                                     stratify = Y)

# Fitting logistic regression to the training set
logisticmodel = sm.Logit(y_train, x_train).fit()


# Evaluate on train dfata
y_pred_train = logisticmodel.predict(x_train)
y_pred_train



#Metrix
#Filling all the cells with zeros

y_train["pred"] = np.zeros(len(y_train))

#taking threshold value and above the prob value will be treated as correct
y_train.loc[pred > optimal_threshold, "pred"] = 1

auc = metrics.roc_auc_score(y_train.Clicked_on_Ad, y_pred_train)
print("Area under the ROC curve : %f" % auc)

classification_train = classification_report(y_train["pred"],y_train.Clicked_on_Ad)
print(classification_train)

# confusion matrix

confusion_matrix(y_train["pred"], y_train.Clicked_on_Ad)
      

# # Taking threshold value and above thr prob value will be treated as correction

# y_train.loc[pred >optimal_threshold, "pred"] = 1

# auc = metrics.roc_auc_score(y_train["ATTORNEY"], y_pred_train)
# print("Area under the ROC curve : %f" % auc)

# classification_train = classification_report(y_train["ATTORNEY"])

# accuracy score of the model
print('Test accuracy = ', accuracy_score(y_train["pred"], y_train.Clicked_on_Ad))

# Validation on test data

y_pred_test = logisticmodel.predict(x_test)
y_pred_test


# filling all the cells with zeroes
y_test["y_pred_test"] = np.zeros(len(y_test))

#capturing the prediction binary values
y_test.loc[y_pred_test > optimal_threshold, "y_pred_test"] = 1

# classification report
classification1 = classification_report(y_test["y_pred_test"], y_test.Clicked_on_Ad)
print(classification1)


# confusion matrix
confusion_matrix(y_test["y_pred_test"],y_test.Clicked_on_Ad)

#Accuracy score of the model
print('Test accuracy = ', accuracy_score(y_test["y_pred_test"], y_test.Clicked_on_Ad))

# test the best model on new data











