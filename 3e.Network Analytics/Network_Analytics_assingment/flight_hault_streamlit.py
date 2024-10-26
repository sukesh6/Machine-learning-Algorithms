

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx

# Streamlit App Title
st.title("Network Analysis of Connecting Routes")

# Load the CSV file
@st.cache_data
def load_data():
    return pd.read_csv(r"C:/Users/sukes/OneDrive/Desktop/360/ds Assingments/3e.Network Analytics/Datasets/flight_hault.csv")

H_D = load_data()

# Decrease rows and columns
H_D = H_D.iloc[:350, 1:6]

# Create the graph
G = nx.Graph()
G = nx.from_pandas_edgelist(H_D, source='IATA_FAA', target='ICAO')



# Calculate centralities
centralities = pd.DataFrame({
    'Degree': pd.Series(nx.degree_centrality(G)),
    'Closeness': pd.Series(nx.closeness_centrality(G)),
    'Betweenness': pd.Series(nx.betweenness_centrality(G)),
    'Eigenvector': pd.Series(nx.eigenvector_centrality(G, max_iter=500))
})

st.write("### Centralities")
st.write(centralities)

# Draw the network graph
st.write("### Network Graph Visualization")

fig, ax = plt.subplots(figsize=(10, 10))
pos = nx.spring_layout(G, k=0.10)
nx.draw_networkx(G, pos, node_size=15, node_color='red', ax=ax)
plt.title("Network Graph of Connecting Routes")
st.pyplot(fig)

