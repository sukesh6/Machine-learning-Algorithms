import pandas as pd

df=pd.read_csv(r'C:/Users/sukes/Downloads/Data Set (1)/Data Set (5)/AirTraffic_Passenger_Statistics.csv')

df.dtypes
df_new=df.drop(['Terminal','Operating Airline IATA Code'],axis=1)

import sweetviz
report=sweetviz.analyze([df_new,'df_new'])
report.show_html('report.html')

#EDA report highlights

import matplotlib.pyplot as plt
df_new.plot(kind='box',subplots=True,sharey=False,figsize=(15,8))
plt.subplots_adjust(wspace=0.75)
plt.show()

#Data Preprocessing 
from AutoClean import AutoClean
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer
from feature_engine.outliers import Winsorizer

clean_pipeline=AutoClean(df_new,
                         mode='manual',
                         missing_num='auto',
                         outliers='winz',
                         encode_categ=['auto'])
df_clean=clean_pipeline.output

df_clean.plot(kind='box',sharey=False,subplots=True,figsize=(15,8))
plt.subplots_adjust(wspace=0.75)
plt.show()


#Normalization/MinMaxScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import make_pipeline
df_clean.info()

cols=list(df_clean.columns)
cols

numeric_features=df_clean.select_dtypes(exclude=['object','bool']).columns
numeric_features

categorical_features=df_clean.select_dtypes(include=['object']).columns
categorical_features

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
column_transformer=ColumnTransformer(transformers=[('encoder',OneHotEncoder(sparse_output=False),
                                                    categorical_features)],
                                     remainder='passthrough')


pipe1=make_pipeline(column_transformer)

df_pipelined=pd.DataFrame(pipe1.fit_transform(df_clean),columns=pipe1.get_feature_names_out())

#Model Building

from scipy.cluster.hierarchy import linkage,dendrogram
from sklearn.cluster import AgglomerativeClustering

#dendrogram

plt.figure(1,figsize=(16,8))
tree_plot=dendrogram(linkage(df_pipelined,method='complete'))
plt.title('Hierarchical clustering dendrogram')
plt.xlabel('index')
plt.ylabel('Euclidean Distence')
plt.show

hc1=AgglomerativeClustering(n_clusters=3,metric='euclidean',linkage='complete')
y_hc1=hc1.fit_predict(df_pipelined)
y_hc1
hc1.labels_

cluster_labels=pd.Series(hc1.labels_)

df_clust=pd.concat([cluster_labels,df_new],axis=1)
df_clust=df_clust.rename(columns={0:'clusters'})

#Cluster Evaluation

from sklearn import metrics
metrics.silhouette_score(df_pipelined,cluster_labels)
metrics.calinski_harabasz_score(df_pipelined,cluster_labels)
metrics.davies_bouldin_score(df_pipelined,cluster_labels)

#HyperParameter optimization for H clustering

import numpy as np
from clusteval import clusteval

ce=clusteval(evaluate='silhouette')
df_array=np.array(df_pipelined)
ce.fit(df_array)
ce.plot()

hc2_clust=AgglomerativeClustering(n_clusters=2,metric='euclidean',linkage='ward')
y_hc2_clust=hc2_clust.fit_predict(df_pipelined)
hc2_clust.labels_
cluster2_labels=pd.Series(hc2_clust.labels_)
df_clust2=pd.concat([cluster2_labels,df_new],axis=1)
df_clust2=df_clust2.rename(columns={0:'clusters'})

metrics.silhouette_score(df_pipelined,cluster2_labels)
metrics.calinski_harabasz_score(df_pipelined,cluster2_labels)
metrics.davies_bouldin_score(df_pipelined,cluster2_labels)

#___________________________________________________________________________________________

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sweetviz
from AutoClean import AutoClean
from sklearn.preprocessing import MinMaxScaler 
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import make_pipeline
from scipy.cluster.hierarchy import linkage, dendrogram
from sklearn import metrics
from clusteval import clusteval
from feature_engine.outliers import Winsorizer
from sklearn.compose import ColumnTransformer
from sqlalchemy import create_engine, text 

df=pd.read_csv(r"C:/Users/sukes/Downloads/Data Set (1)/Data Set (5)/AirTraffic_Passenger_Statistics.csv")

from sqlalchemy import create_engine,text
from urllib.parse import quote

user='root'
pw=quote('bunny@86882')
db='datascience'

engine=create_engine(f'mysql+pymysql://{user}:{pw}@localhost/{db}')
df.to_sql('airline_tbl',con=engine,if_exists='replace',chunksize=1000,index=False)

sql=('select * from airline_tbl;')

airline=pd.read_sql_query(sql,engine.connect())



import sweetviz
report=sweetviz.analyze([airline,'airline'])
report.show_html('Report.html')

airline_new=airline.drop(['Terminal','Operating Airline IATA Code'],axis=1)

report1=sweetviz.analyze([airline_new,'airline_new'])
report.show_html('report_new_html')

#EDA Report Highlights
import matplotlib.pyplot as plt

airline_new.plot(kind='box',subplots=True,sharey=False,figsize=(15,8))
plt.subplots_adjust(wspace=0.75)
plt.show()

#After visuvalizing the box plot Passenger count has high number of outliers so we need to winosrize that

from feature_engine.outliers import Winsorizer
winsor=Winsorizer(capping_method='iqr',
                  tail='both',
                  fold=1.5,
                  variables=['Passenger Count'])

airline_new['Passenger Count']=pd.DataFrame(winsor.fit_transform(airline_new[['Passenger Count']]))

airline_new.plot(kind='box',subplots=True,sharey=False,figsize=(15,8))

#_______________________Data Preprocessing______________________________________________

from AutoClean import AutoClean

clean_pipeline=AutoClean(airline_new.iloc[:,:],
                         mode='manual',
                         missing_num='auto',
                         outliers='winz',
                         encode_categ=['auto'])

airline_clean=clean_pipeline.output

airline_clean.head()
airline_clean.info()
airline_clean.dtypes

categorical_columns=airline_clean.select_dtypes(include=['object']).columns
categorical_columns

numerical_columns=airline_clean.select_dtypes(exclude=['object','bool']).columns
numerical_columns

from sklearn.compose import ColumnTransformer
column_transformer = ColumnTransformer(transformers = [('encoder', OneHotEncoder(sparse_output = False), categorical_columns)], remainder = 'passthrough')

pipe1 = make_pipeline(column_transformer)

airline_pipelined = pd.DataFrame(pipe1.fit_transform(airline_new), columns = pipe1.get_feature_names_out())

#for dendrogram

plt.figure(1, figsize = (16, 8))
tree_plot = dendrogram(linkage(airline_pipelined, method =  "ward"))
plt.title('Hierarchical Clustering Dendrogram')
plt.xlabel('Index')
plt.ylabel('Euclidean distance')
plt.show()

#Dendrogram gives 2 clusters

from sklearn.cluster import AgglomerativeClustering
hc1 = AgglomerativeClustering(n_clusters = 2, metric = 'euclidean', linkage = 'complete')

y_hc1 = hc1.fit_predict(airline_pipelined)
y_hc1

hc1.labels_
cluster_labels = pd.Series(hc1.labels_)
airline_clust = pd.concat([cluster_labels, airline_new], axis = 1)
airline_clust.head()
airline_clust.columns
airline_clust = airline_clust.rename(columns = {0 : 'cluster'})

metrics.silhouette_score(airline_pipelined, cluster_labels)

from clusteval import clusteval
ce = clusteval(evaluate = 'silhouette')
df_array = np.array(airline_pipelined)
ce.fit(df_array)
ce.plot()

df_3clust = pd.concat([df, airline_clust], axis = 1)
df_3clust
df_3clust = df_3clust.rename(columns = {0 : 'cluster'})
df_3clust.to_csv('Airlines_clust.csv', encoding = 'utf-8')



















