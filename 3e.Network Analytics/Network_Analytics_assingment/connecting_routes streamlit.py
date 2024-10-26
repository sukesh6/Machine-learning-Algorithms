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
    return pd.read_csv(r"C:/Users/sukes/OneDrive/Desktop/360/ds Assingments/3e.Network Analytics/Datasets/connecting_routes.csv")

C_R = load_data()

# Decrease rows and columns
C_R = C_R.iloc[:150, 1:6]

# Create the graph
graph = nx.Graph()
graph = nx.from_pandas_edgelist(C_R, source='AER', target='KZN')



# Calculate centralities
centralities = pd.DataFrame({
    'Degree': pd.Series(nx.degree_centrality(graph)),
    'Closeness': pd.Series(nx.closeness_centrality(graph)),
    'Betweenness': pd.Series(nx.betweenness_centrality(graph)),
    'Eigenvector': pd.Series(nx.eigenvector_centrality(graph, max_iter=500))
})

st.write("### Centralities")
st.write(centralities)

# Draw the network graph
st.write("### Network Graph Visualization")

fig, ax = plt.subplots(figsize=(10, 10))
pos = nx.spring_layout(graph, k=0.10)
nx.draw_networkx(graph, pos, node_size=15, node_color='red', ax=ax)
plt.title("Network Graph of Connecting Routes")
st.pyplot(fig)

# Show Streamlit app run command
st.write("Run the app using `streamlit run your_script_name.py`")
