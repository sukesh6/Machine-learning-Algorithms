import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx

# Streamlit App Title
st.title("Network Visualization for Facebook, Instagram, and LinkedIn")

# Load datasets
@st.cache_data
def load_data():
    facebook = pd.read_csv(r"C:/Users/sukes/OneDrive/Desktop/360/ds Assingments/3e.Network Analytics/Datasets/facebook.csv")
    instagram = pd.read_csv(r"C:/Users/sukes/OneDrive/Desktop/360/ds Assingments/3e.Network Analytics/Datasets/instagram.csv")
    linkedin = pd.read_csv(r"C:/Users/sukes/OneDrive/Desktop/360/ds Assingments/3e.Network Analytics/Datasets/linkedin.csv")
    return facebook, instagram, linkedin

facebook, instagram, linkedin = load_data()

# Sidebar for dataset selection
dataset_choice = st.sidebar.selectbox("Choose a dataset", ("Facebook", "Instagram", "LinkedIn"))

# Function to create a network graph
def draw_network(data, layout_type):
    g = nx.Graph(data.values)
    pos = nx.circular_layout(g) if layout_type == 'Circular' else nx.spring_layout(g)
    fig, ax = plt.subplots(figsize=(8, 8))
    nx.draw_networkx(g, pos, node_size=50, node_color='skyblue', ax=ax)
    plt.title(f"{layout_type} Network Visualization")
    st.pyplot(fig)

# Network type selection
layout_type = st.radio("Select network layout", ('Circular Network', 'Star Network'))

# Displaying the chosen dataset and graph
if dataset_choice == "Facebook":
    st.write(" Facebook Network Data")
    st.write(facebook.head())  # Display first few rows
    st.write(" Facebook Network Graph")
    draw_network(facebook, layout_type)
elif dataset_choice == "Instagram":
    st.write(" Instagram Network Data")
    st.write(instagram.head())  # Display first few rows
    st.write(" Instagram Network Graph")
    draw_network(instagram, layout_type)
else:
    st.write("LinkedIn Network Data")
    st.write(linkedin.head())  # Display first few rows
    st.write(" LinkedIn Network Graph")
    draw_network(linkedin, layout_type)

