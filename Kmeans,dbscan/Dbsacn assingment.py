import pandas as pd
import matplotlib.pyplot as plt
import sweetviz

from sklearn.cluster import KMeans,AgglomerativeClustering,DBSCAN
from sklearn.metrics import silhouette_score

df=pd.read_csv(r"C:/Users/sukes/Downloads/Data Set (1)/Data Set (5)/AirTraffic_Passenger_Statistics.csv")

my_report=sweetviz.analyze([df,'df'])
my_report.show_html('report.html')

df.info()
df.describe()

duplicate=df.duplicated()
duplicate
duplicate.sum()
df=df.drop_duplicates()

print(df.shape)
df.columns

#Take only columns of Operating Airline, Geo Region, Passanger Count
df1=df[['Operating Airline','GEO Region','Passenger Count']]

airline_count=df1['Operating Airline'].value_counts()
airline_count.sort_index(inplace=True)

passenger_count=df1.groupby('Operating Airline').sum()['Passenger Count'] 
passenger_count.sort_index(inplace=True)

df2=pd.concat([airline_count,passenger_count],axis=1)
plt.figure(figsize = (10,10))
plt.scatter(df2['count'], df2['Passenger Count'])
plt.xlabel("Flights held")
plt.ylabel("Passengers")
for i, txt in enumerate(airline_count.index.values):
    a = plt.gca()
    plt.annotate(txt, (df2['count'][i], df2['Passenger Count'][i]))
plt.show()

df2.index

index_labels_to_drop = ['United Airlines', 'United Airlines - Pre 07/01/2013']
df3 = df2.drop(index_labels_to_drop)

ac=AgglomerativeClustering(2,linkage='ward')
ac_clusters=ac.fit_predict(df3)

km=KMeans(n_clusters=2)
km_clusters=ac.fit_predict(df3)

db_param_options = [[8000000, 2], [7500000, 2], [8200000, 2], [6800000, 3], [6500000, 3], [6000000, 3]]

for ep, min_sample in db_param_options:
    db = DBSCAN(eps = ep, min_samples = min_sample)
    db_clusters = db.fit_predict(df3)
    print("Eps: ", ep, "Min Samples: ", min_sample)
    print("DBSCAN Clustering: ", silhouette_score(df3, db_clusters))

db = DBSCAN(eps = 8000000, min_samples = 2)
db_clusters = db.fit_predict(df3)

plt.figure(1)
plt.title("Airline Clusters from Agglomerative Clustering")
plt.scatter(df3['count'], df3['Passenger Count'], c = ac_clusters, s = 50, cmap = 'tab20b')
plt.show()

plt.figure(2)
plt.title("Airline Clusters from K-Means")
plt.scatter(df3['count'], df3['Passenger Count'], c = km_clusters, s = 50, cmap = 'tab20b')
plt.show()

plt.figure(3)
plt.title("Airline Clusters from DBSCAN")
plt.scatter(df3['count'], df3['Passenger Count'], c = db_clusters, s = 50, cmap = 'tab20b')
plt.show()

print("Silhouette Scores for Wine Dataset:\n")

print("Agg Clustering: ", silhouette_score(df3, ac_clusters))

print("K-Means Clustering: ", silhouette_score(df3, km_clusters))

print("DBSCAN Clustering: ", silhouette_score(df3, db_clusters))

import pickle
pickle.dump(db, open('db.pkl', 'wb'))

model = pickle.load(open('db.pkl', 'rb'))

res = model.fit_predict(df3)

#---------------------------------------------------------------------------------------------
import pandas as pd
import sweetviz
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans,AgglomerativeClustering,DBSCAN
from sklearn.metrics import silhouette_score
from sqlalchemy import create_engine,text
from sklearn.compose import ColumnTransformer
from feature_engine.outliers import Winsorizer
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

df=pd.read_csv(r"C:/Users/sukes/Downloads/Data Set (1)/Data Set (5)/AirTraffic_Passenger_Statistics.csv")

df.dtypes
df_new=df.drop(['Activity Period','Operating Airline IATA Code','Terminal'],axis=1)

import sweetviz
my_report=sweetviz.analyze([df_new,'df_new'])
my_report.show_html('my_report.html')

#Data Preprocessing and cleaning

df_new.info()

duplicate=df_new.duplicated()
duplicate

sum(duplicate)
df_new=df_new.drop_duplicates()

#Missing Value Analysis
df_new.isnull().sum()

#Outlier Analysis
import matplotlib.pyplot as plt
df_new.plot(kind='box',subplots=True,sharey=False,figsize=(16,8))
plt.subplots_adjust(wspace=0.75)
plt.show()

from feature_engine.outliers import Winsorizer
winsor=Winsorizer(capping_method='iqr',
                  tail='both',
                  fold=1.5,
                  variables=['Passenger Count'])

df_new['Passenger Count']=pd.DataFrame(winsor.fit_transform(df_new[['Passenger Count']]))

df_new.plot(kind='box',sharey=False,subplots=True,figsize=(16,8))
plt.subplots_adjust(wspace=0.75)
plt.show()

numeric_features=df_new.select_dtypes(exclude=['bool','object']).columns
numeric_features
num_pipeline=Pipeline([('impute',SimpleImputer(strategy='mean')),
                       ('sacle',MinMaxScaler())])
categorical_features=df_new.select_dtypes(include=['object']).columns
categorical_features
categ_pipeline=Pipeline([('encoder',OrdinalEncoder())])

from sklearn.compose import ColumnTransformer

process_pipeline=ColumnTransformer([('categorical',categ_pipeline,categorical_features),
                                    ('numerical',num_pipeline,numeric_features)],
                                   remainder='passthrough')
process_pipeline
processed=process_pipeline.fit(df_new)
processed
df_clean=pd.DataFrame(processed.transform(df_new),columns=processed.get_feature_names_out())


#Clusters (Using AgglomerativeClustering)
from sklearn.cluster import KMeans,AgglomerativeClustering,DBSCAN
ac=AgglomerativeClustering(5,linkage='average')
ac_clust=ac.fit_predict(df_clean)
ac_clust

#Kmeans Clustering

km=KMeans(5)
km_clusters=km.fit_predict(df_clean)
km_clusters

#DBSCAN Clustering
db_parm_options=[[6,9]]
for ep,min_sample in db_parm_options:
    db=DBSCAN(eps=ep,min_samples=min_sample)
    db_clusters=db.fit_predict(df_clean)
    print('eps: ',ep,'min_samples: ',min_sample)
    print('DBSCAN Clustering:',silhouette_score(df_clean,db_clusters))
    
db=DBSCAN(eps=6,min_samples=9)
db_clusters=db.fit_predict(df_clean)
db_clusters

#Calculate Silhoutte score

print('Silhouette score for wine dataset:\n')
print('Agglomerative clustering:',silhouette_score(df_clean,ac_clust))
print('KMeans clustering:',silhouette_score(df_clean,km_clusters))
print('DBSCAN clustering:',silhouette_score(df_clean,km_clusters))

#___________________________________________________________________________________________________________













