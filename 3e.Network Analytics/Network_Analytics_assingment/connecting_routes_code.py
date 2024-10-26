import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

connecting_routes=pd.read_csv(r"C:/Users/sukes/OneDrive/Desktop/360/ds Assingments/3e.Network Analytics/Datasets/connecting_routes.csv")

connecting_routes.head()
connecting_routes.describe()
connecting_routes.columns
connecting_routes.info()

#Dropping the columns has zero varience
con_routes=connecting_routes.drop(['Unnamed: 6','0','CR2'],axis=1)

##creating dataframe with 150 rows to visivualize graph effectively
routes=con_routes.iloc[:150,:]
routes.columns

#creating graph
graph=nx.Graph()
graph=nx.from_pandas_edgelist(routes,source='AER',target='KZN')
print(graph)
print("number of nodes:",len(graph.nodes))
print('number of edges:',len(graph.edges))

#Create centrality measures
centrality_measures=pd.DataFrame({
    "closeness":pd.Series(nx.closeness_centrality(graph)),
    "Degree":pd.Series(nx.degree_centrality(graph)),
    "eigenvector":pd.Series(nx.eigenvector_centrality(graph,max_iter=300)),
    "betweeness":pd.Series(nx.betweenness_centrality(graph)),
    "cluster_coefficent":pd.Series(nx.clustering(graph))})

fig=nx.spring_layout(graph,k=0.015)
nx.draw_networkx(graph,fig,node_size=15,node_color='red')
plt.show()