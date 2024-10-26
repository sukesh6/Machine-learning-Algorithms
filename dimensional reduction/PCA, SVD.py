import pandas as pd
import numpy as np
import sweetviz
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.decomposition import PCA
import joblib
from kneed import KneeLocator
from sqlalchemy import create_engine
from urllib.parse import quote

df=pd.read_csv(r"C:/Users/sukes/Downloads/Data Set (1)/Data Set (5)/AutoInsurance.csv")
df.info()

#To analyze dat by using of autoeda
my_report=sweetviz.analyze(df)
my_report.show_html('report.html')

df1=df.drop(['EmploymentStatus','Policy Type','Monthly Premium Auto','Number of Policies'],axis=1)
df1.isnull().sum()
df1.duplicated().sum()
df1.info()
df1['Customer Lifetime Value']=df1['Customer Lifetime Value'].astype('int')
df1['Total Claim Amount']=df1['Total Claim Amount'].astype('int')

numeric_features=df1.select_dtypes(exclude=['object']).columns
numeric_features

categorical_features=df1.select_dtypes(include=['object']).columns
categorical_features

#Defining PCA with 6 components
pca=PCA(n_components=6)

#Pipeline
num_pipeline=make_pipeline(SimpleImputer(strategy='mean'),StandardScaler(),pca)
num_pipeline

#Fit the numeric features in Numerical Pipeline
processed=num_pipeline.fit(df1[numeric_features])
processed

#Applying pipeline on orginal dataset to transform it using imputation,std,pca
clean_data=pd.DataFrame(processed.transform(df1[numeric_features]))
clean_data

#Saving the model
joblib.dump(processed,'data_prep_dimenssionalredection')
import os
os.getcwd

#loading the saved pipeline model
model=joblib.load('data_prep_dimenssionalredection')

#Applying the saved model on the dataset to extract pca values
pca_res=pd.DataFrame(model.transform(df1[numeric_features]))
pca_res

#Getting pca weight components from the saved model
model['pca'].components_

#Storing the pca weight components in a dataset
components=pd.DataFrame(model['pca'].components_,columns=numeric_features).T
components.columns=['pc0','pc1','pc2','pc3','pc4','pc5']
components

#Printing the varience
var=model['pca'].explained_variance_ratio_
sum(var)
print(var)

#Calculating the cummulative sum
var1=np.cumsum(model['pca'].explained_variance_ratio_)
var1

#Plotting the varience using by pca components
plt.plot(var1,color='red')

#KneeLocator
k1=KneeLocator(range(len(var1)),var1,curve='concave',direction='increasing')
k1.elbow
plt.style.use('ggplot')
plt.plot(range(len(var1)), var1)
plt.xticks(range(len(var1)))
plt.ylabel('varience')
plt.xlabel('number of principal components')
plt.axvline(x=k1.elbow,color='r',label='Elbow Point',ls='--')
plt.show()

#Concating the droped columns
final=pd.concat([df['EmploymentStatus'],df['Policy Type'],df['Monthly Premium Auto'],df['Number of Policies'],pca_res.iloc[:,0:5]],axis=1)
final.columns=['EmploymentStatus','Policy Type','Monthly Premium Auto','Number of Policies','pc0','pc1','pc2','pc3','pc4']

#creatting a scatter plot of pc0 vs pc1 
ax=final.plot(x='pc0',y='pc1',kind='scatter',figsize=(12,8))
final[['pc0','pc1','EmploymentStatus']].apply(lambda x:ax.text(*x),axis=1)


#-------------------------------SVD-------------------------------------------

import pandas as pd
import numpy as np
import sweetviz
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OrdinalEncoder
from sklearn.pipeline import make_pipeline
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA,TruncatedSVD
import joblib
from kneed import KneeLocator
from sqlalchemy import create_engine
from urllib.parse import quote

df=pd.read_csv(r"C:/Users/sukes/Downloads/Data Set (1)/Data Set (5)/AutoInsurance.csv")

my_report=sweetviz.analyze(df)
my_report.show_html('report.html')

df1=df.drop(['EmploymentStatus','Vehicle Class','Policy Type'],axis=1)
df1.isnull().sum()
df.duplicated().sum()
df.info()

numeric_features=df1.select_dtypes(exclude=['object']).columns
numeric_features

categorical_features=df1.select_dtypes(include=['object']).columns
categorical_features 

svd=TruncatedSVD(n_components=21)

pipeline=Pipeline(steps=[('encoding',OrdinalEncoder()),
                 ('scale',StandardScaler()),
                 ('svd',svd)])

processed=pipeline.fit(df1)
processed

clean_data=pd.DataFrame(processed.transform(df1),columns=df1.columns)
clean_data

joblib.dump(processed,'svd_ass')

model=joblib.load('svd_ass')
model

svd_res=pd.DataFrame(model.transform(df1))
svd_res

model['svd'].components_

components=pd.DataFrame(model['svd'].components_,columns=df1.columns).T
components.columns=['pc0','pc1','pc2','pc3','pc4','pc5','pc6','pc7','pc8','pc9','pc10','pc11','pc12','pc13','pc14','pc15','pc16','pc17','pc18','pc19','pc20']
components

print(model['svd'].explained_variance_ratio_)

var1=np.cumsum(model['svd'].explained_variance_ratio_)
var1

plt.plot(var1,color='red')


k1=KneeLocator(range(len(var1)),var1,curve='concave',direction='increasing')
k1.elbow
plt.style.use('ggplot')
plt.plot(range(len(var1)), var1)
plt.xticks(range(len(var1)))
plt.ylabel('varience')
plt.xlabel('number of principal components')
plt.axvline(x=k1.elbow,color='r',label='Elbow Point',ls='--')
plt.show()

#knee locator gives 0 - 14 columns with 82% data
