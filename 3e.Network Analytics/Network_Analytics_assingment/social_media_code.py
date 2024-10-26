import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

fb=pd.read_csv(r'C:/Users/sukes/OneDrive/Desktop/360/ds Assingments/3e.Network Analytics/Datasets/facebook.csv')
insta=pd.read_csv(r"C:/Users/sukes/OneDrive/Desktop/360/ds Assingments/3e.Network Analytics/Datasets/instagram.csv")
linkedin=pd.read_csv(r"C:/Users/sukes/OneDrive/Desktop/360/ds Assingments/3e.Network Analytics/Datasets/linkedin.csv")

#●	circular network for Facebook
fb_circular = nx.cycle_graph(6)  # 6 users in a circle

# Draw the network
plt.figure(figsize=(8, 6))
nx.draw(fb_circular, with_labels=True, node_color='lightblue', node_size=700, font_size=16, font_color='black')
plt.title('Circular Network for Facebook', fontsize=18)
plt.show()

#●	star network for Instagram
instagram_star = nx.star_graph(5)  # 1 central node and 5 outer nodes

# Draw the network
plt.figure(figsize=(8, 6))
nx.draw(instagram_star, with_labels=True, node_color='lightcoral', node_size=700, font_size=16, font_color='black')
plt.title('Star Network for Instagram', fontsize=18)
plt.show()

#●	star network for LinkedIn
linkedin_star=nx.star_graph(4)
plt.figure(figsize=(8,6))
nx.draw(linkedin_star, with_labels=True, node_color='lightgreen', node_size=700, font_size=16, font_color='black')
plt.title('Star Network for LinkedIn', fontsize=18)
plt.show()