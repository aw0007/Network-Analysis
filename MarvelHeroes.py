
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import os

# === File paths ===
edges_path = r"C:\Users\...\The Marvel Universe Social Network\hero-network.csv"
nodes_path = r"C:\Users\....\The Marvel Universe Social Network\nodes.csv"

# === Load data ===
edges = pd.read_csv(edges_path)
nodes = pd.read_csv(nodes_path)
edges.columns = ['hero1', 'hero2']

# === Count hero appearances
hero_counts = pd.concat([edges['hero1'], edges['hero2']]).value_counts()
top_30_heroes = hero_counts.head(30).index.tolist()

# === Filter edges for top 30
filtered_edges = edges[
    edges['hero1'].isin(top_30_heroes) & edges['hero2'].isin(top_30_heroes)
]

# === Build graph
G = nx.Graph()
G.add_edges_from(zip(filtered_edges['hero1'], filtered_edges['hero2']))

# === Compute PageRank centrality
pagerank = nx.pagerank(G)
scaler = MinMaxScaler(feature_range=(300, 1200))
sizes = scaler.fit_transform([[pagerank[n]] for n in G.nodes()]).flatten()

# === Assign top 5 colors
top_5 = sorted(pagerank.items(), key=lambda x: x[1], reverse=True)[:5]
top_5_heroes = [hero for hero, _ in top_5]

custom_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']  # blue, orange, green, red, purple
hero_colors = {hero: color for hero, color in zip(top_5_heroes, custom_colors)}
node_colors = [hero_colors.get(n, "#d3d3d3") for n in G.nodes()]  # light gray default

# === Layout and plot
pos = nx.spring_layout(G, seed=42)

plt.figure(figsize=(11, 7))
nx.draw_networkx_edges(G, pos, alpha=0.15, edge_color="gray", width=1.2)
nx.draw_networkx_nodes(G, pos, node_size=sizes, node_color=node_colors, edgecolors="white", linewidths=0.6)
nx.draw_networkx_labels(G, pos, font_size=7, font_family="sans-serif")

plt.title("🦸 Top 30 Marvel Heroes Network", fontsize=14, fontweight='bold', loc='left')
plt.axis("off")

# === Save output
output_path = r"C:\Users\massa\Desktop\MiniProjet\Networks analysis\The Marvel Universe Social Network\marvel_top30_style.png"
plt.savefig(output_path, dpi=300, bbox_inches="tight")
plt.show()
print(f"📸 Saved styled graph to: {output_path}")
