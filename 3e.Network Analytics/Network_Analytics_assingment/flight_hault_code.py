import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from sqlalchemy import create_engine
from urllib.parse import quote
from geopy.geocoders import Nominatim

flight=pd.read_csv(r"C:/Users/sukes/OneDrive/Desktop/360/ds Assingments/3e.Network Analytics/Datasets/flight_hault.csv")

flight.head()

#creating dataframe with 350 rows to visivualize graph effectively
flight_routes=flight.iloc[:100,:13]
flight_routes

#Creating empty graph
empty_graph=nx.Graph()
empty_graph=nx.from_pandas_edgelist(flight_routes,source="IATA_FAA",target='ICAO')
print(empty_graph)
print("number of nodes:",len(empty_graph.nodes))
print('number of edges:',len(empty_graph.edges))

#creating centarility measures
centrality_measures=pd.DataFrame({
    "closeness":pd.Series(nx.closeness_centrality(empty_graph)),
    "Degree":pd.Series(nx.degree_centrality(empty_graph)),
    "eigenvector":pd.Series(nx.eigenvector_centrality(empty_graph,max_iter=300)),
    "betweeness":pd.Series(nx.betweenness_centrality(empty_graph)),
    "cluster_coefficent":pd.Series(nx.clustering(empty_graph))})

fig=nx.spring_layout(empty_graph,k=0.015)
nx.draw_networkx(empty_graph,fig,node_size=15,node_color='red')
plt.show()