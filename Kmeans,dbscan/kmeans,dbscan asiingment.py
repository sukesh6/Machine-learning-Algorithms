import pandas as pd
import sweetviz
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.cluster import KMeans
from sklearn import metrics
import joblib
import pickle
from sqlalchemy import create_engine,text
from urllib.parse import quote

df=pd.read_csv(r"C:/Users/sukes/Downloads/Data Set (1)/Data Set (5)/AirTraffic_Passenger_Statistics.csv")

df.shape
df.dtypes

df_new=df.drop(['Operating Airline IATA Code','Terminal','Activity Period'],axis=1)

report=sweetviz.analyze([df_new,'df_new'])
report.show_html('repoer.html')

#Discriptive Analysis
df_new.describe()
df_new.info()

#To check any missing data
missing_data=df_new.isnull().sum()
missing_data

#EDA Report analysis
df_new.plot(kind='box',sharey=False,subplots=True,figsize=(15,8))
plt.subplots_adjust(wspace=0.75)
plt.show()

#Data preprocessing
from AutoClean import AutoClean

clean_pipeline=AutoClean(df_new,
                         mode='manual',
                         missing_num='auto',
                         outliers='winz',
                         encode_categ=['auto'])
df_clean=clean_pipeline.output

df_clean.plot(kind='box',sharey=False,subplots=True,figsize=(15,8))
plt.subplots_adjust(wspace=0.75)
plt.show()

numerical_features=df_new.select_dtypes(exclude=['object','bool']).columns
numerical_features
from sklearn.pipeline import Pipeline
num_pipeline=Pipeline([('impute',SimpleImputer(strategy='mean')),
                       ('scale',MinMaxScaler())])
num_pipeline

categorical_features=df_new.select_dtypes(include=['object']).columns
categorical_features

categ_pipeline=Pipeline([('encoder',OrdinalEncoder())])
categ_pipeline

from sklearn.compose import ColumnTransformer

process_pipeline=ColumnTransformer([('categorical',categ_pipeline,categorical_features),
                                    ('numerical',num_pipeline,numerical_features)],
                                   remainder='passthrough')
process_pipeline

processed=process_pipeline.fit(df_new)

import joblib
joblib.dump(processed,'preprocessing')

airdata_clean=pd.DataFrame(processed.transform(df_new),columns=processed.get_feature_names_out())

#Clustering and Model Building
from sklearn.cluster import KMeans

TWSS=[]
k=list(range(2,9))
for i in k:
    kmeans=KMeans(n_clusters=i)
    kmeans.fit(airdata_clean)
    TWSS.append(kmeans.inertia_)
TWSS

#Creating a screeplot

plt.plot(k,TWSS,'ro-')
plt.xlabel('No of clusters')
plt.ylabel('Total within ss')

#Using Kneelocator

list=[]
for k in range(2,9):
    kmeans=KMeans(n_clusters=k)
    kmeans.fit(airdata_clean)
    list.append(kmeans.inertia_)
list    

#!pip install kneed
from kneed import KneeLocator
k1=KneeLocator(range(2,9),list,curve='convex',direction='decreasing')
k1.elbow
plt.style.use("ggplot")
plt.plot(range(2,9),list)
plt.xticks(range(2,9))
plt.ylabel('Inertia')
plt.axvline(x=k1.elbow,color='r',label='axvline-full height',ls='--')
plt.show()

#Kmeans model building with 3 clusters
model=KMeans(n_clusters=3)
yy=model.fit(airdata_clean)
cluster_labels=model.labels_
cluster_labels
tr=pd.Series(cluster_labels)
#Cluster Evaluation
from sklearn import metrics
silhouette_score=metrics.silhouette_score(airdata_clean,cluster_labels)
silhouette_score

from sklearn.metrics import silhouette_score

silhouette_coefficients=[]
for k in range(2,9):
    kmeans=KMeans(n_clusters=k)
    kmeans.fit(airdata_clean)
    score=silhouette_score(airdata_clean,kmeans.labels_)
    silhouette_coefficients.append([k,score])
silhouette_coefficients   

sorted(silhouette_coefficients,reverse=True,key=lambda x:x[1])

#Building Kmeans clustering

bestmodel=KMeans(n_clusters=2)
result=bestmodel.fit(airdata_clean)

import pickle
pickle.dump(result,open('KmeansClust_airdata.pkl','wb'))

cluster_labels1=bestmodel.labels_
cluster_labels1
labels=pd.Series(cluster_labels1)

silhouette_score=metrics.silhouette_score(airdata_clean,cluster_labels1)
silhouette_score #Heer I got score as .6296310059460033

df_clust=pd.concat([labels,df],axis=1)
df_clust=df_clust.rename(columns={0:'clusters'})

clust_agg= df_clust["Passenger Count"].groupby(df_clust.clusters).mean()
clust_agg
df_clust.to_csv('KMeans_Airline.csv',encoding='utf-8',index=False)

#-------------------------------------------------------------------------


import pandas as pd 
import sweetviz
import matplotlib.pyplot as plt


# from sklearn.compose import ColumnTransformer
# from sklearn.preprocessing import OneHotEncoder

from sklearn.cluster import KMeans
from sklearn import metrics

import pickle
# **Import the data**

from sqlalchemy import create_engine

from urllib.parse import quote
df = pd.read_csv(r"C:/Users/sukes/Downloads/Data Set (1)/Data Set (5)/AirTraffic_Passenger_Statistics.csv")

"""# Credentials to connect to Database
user = 'user1' # user name
pw = quote('user1') # password
db = 'air_routes_db' # database
# creating engine to connect MySQL database
engine = create_engine(f"mysql+pymysql://{user}:{pw}@localhost/{db}")

# to_sql() - function to push the dataframe onto a SQL table.
df.to_sql('airline_tbl', con = engine, if_exists = 'replace', chunksize = 1000, index = False)

sql = 'select * from airline_tbl;'
df = pd.read_sql_query(sql, engine)"""

# Data types
df.info()
df.isnull().sum()

# EXPLORATORY DATA ANALYSIS (EDA) / DESCRIPTIVE STATISTICS
# ***Descriptive Statistics and Data Distribution Function***

df.describe()

df.duplicated().sum()

# my_report = sweetviz.analyze([df, "df"])
# my_report.show_html('Report.html')

# As we can see there are multiple columns in our dataset, 
# but for cluster analysis we will use 
# Operating Airline, Geo Region, Passenger Count and Flights held by each airline.
df1 = df[["Operating Airline", "GEO Region", "Passenger Count"]]

airline_count = df1["Operating Airline"].value_counts()
airline_count.sort_index(inplace=True)

passenger_count = df1.groupby("Operating Airline").sum()["Passenger Count"]
passenger_count.sort_index(inplace=True)

'''So as this algorithms is working with distances it is very sensitive to outliers, 
that’s why before doing cluster analysis we have to identify outliers and remove them from the dataset. 
In order to find outliers more accurately, we will build the scatter plot.'''

df2 = pd.concat([airline_count, passenger_count], axis=1)
# x = airline_count.values
# y = passenger_count.values
plt.figure(figsize = (10,10))
plt.scatter(df2['count'], df2['Passenger Count'])
plt.xlabel("Flights held")
plt.ylabel("Passengers")
for i, txt in enumerate(airline_count.index.values):
    a = plt.gca()
    plt.annotate(txt, (df2['count'][i], df2['Passenger Count'][i]))
plt.show()

df2.index
# We can see that most of the airlines are grouped together in the bottom left part of the plot, 
# some are above them, and it has 2 outliers United Airlines and Unites Airlines — Pre 07/01/2013.
# So let’s get rid of them.

index_labels_to_drop = ['United Airlines', 'United Airlines - Pre 07/01/2013']
df3 = df2.drop(index_labels_to_drop)


# # CLUSTERING MODEL BUILDING

# ### KMeans Clustering
# Libraries for creating scree plot or elbow curve 
# from sklearn.cluster import KMeans
# import matplotlib.pyplot as plt

###### scree plot or elbow curve ############

TWSS = []
k = list(range(2, 9))

for i in k:
    kmeans = KMeans(n_clusters = i)
    kmeans.fit(df3)
    TWSS.append(kmeans.inertia_)

TWSS

# ## Creating a scree plot to find out no.of cluster
plt.plot(k, TWSS, 'ro-'); plt.xlabel("No_of_Clusters"); plt.ylabel("total_within_SS")

List = []

for k in range(2, 9):
    kmeans = KMeans(n_clusters=k) 
    kmeans.fit(df3)
    List.append(kmeans.inertia_)

from kneed import KneeLocator
kl = KneeLocator(range(2, 9), List, curve='convex', direction='decreasing')  # Adjust curve type and direction if needed

import matplotlib.pyplot as plt
plt.plot(range(2, 9), List)
plt.xticks(range(2, 9))
plt.ylabel("Inertia")
plt.axvline(x=kl.elbow, color='r', label='Elbow', ls='--')
plt.show()

print("Elbow point:", kl.elbow)
# Not able to detect the best K value (knee/elbow) as the line is mostly linear

# Building KMeans clustering
model = KMeans(n_clusters = 5)
yy = model.fit(df3)

# Cluster labels
model.labels_

# ## Cluster Evaluation

# **Silhouette coefficient:**  
# Silhouette coefficient is a Metric, which is used for calculating 
# goodness of clustering technique and the value ranges between (-1 to +1).
# It tells how similar an object is to its own cluster (cohesion) compared to 
# other clusters (separation).
# A score of 1 denotes the best meaning that the data point is very compact 
# within the cluster to which it belongs and far away from the other clusters.
# Values near 0 denote overlapping clusters.

# from sklearn import metrics
metrics.silhouette_score(df3, model.labels_)

# **Calinski Harabasz:**
# Higher value of CH index means cluster are well separated.
# There is no thumb rule which is acceptable cut-off value.
metrics.calinski_harabasz_score(df3, model.labels_)

# **Davies-Bouldin Index:**
# Unlike the previous two metrics, this score measures the similarity of clusters. 
# The lower the score the better the separation between your clusters. 
# Vales can range from zero and infinity
metrics.davies_bouldin_score(df3, model.labels_)

# ### Evaluation of Number of Clusters using Silhouette Coefficient Technique
from sklearn.metrics import silhouette_score

silhouette_coefficients = []

for k in range (2, 9):
    kmeans = KMeans(n_clusters = k)
    kmeans.fit(df3)
    score = silhouette_score(df3, kmeans.labels_)
    k = k
    Sil_coff = score
    silhouette_coefficients.append([k, Sil_coff])

silhouette_coefficients

sorted(silhouette_coefficients, reverse = True, key = lambda x: x[1])


# silhouette coefficients shows the number of clusters 'k = 2' as the best value

# Building KMeans clustering
bestmodel = KMeans(n_clusters = 2)
result = bestmodel.fit(df3)

# ## Save the KMeans Clustering Model
# import pickle
pickle.dump(result, open('Clust_.pkl', 'wb'))

import os
os.getcwd()

# Cluster labels
bestmodel.labels_

mb = pd.Series(bestmodel.labels_) 
df3['cluster_id'] = mb.values
# Concate the Results with data

# Save the Results to a CSV file
df3.to_csv('Air.csv', encoding = 'utf-8', index = True)
 
import os
os.getcwd()

























































