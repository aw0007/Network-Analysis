import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import os
import random
from sklearn.preprocessing import MinMaxScaler

# ==== File path ====
file_path = r"C:\Users\......\token_transfers.csv"

# ==== Load transfer data ====
print("📥 Loading data...")
df = pd.read_csv(file_path, usecols=["from_address", "to_address"])
df.dropna(inplace=True)

# ==== Count transactions per address ====
all_addresses = pd.concat([df["from_address"], df["to_address"]])
top_addresses = all_addresses.value_counts().head(100).index.tolist()

# ==== Filter edges between top 100 ====
filtered_df = df[
    df["from_address"].isin(top_addresses) & df["to_address"].isin(top_addresses)
]

# ==== Build graph ====
G = nx.DiGraph()
G.add_edges_from(zip(filtered_df["from_address"], filtered_df["to_address"]))

# ==== Count total transactions per node ====
tx_count = all_addresses.value_counts().to_dict()

# ==== Add node attributes ====
for node in G.nodes():
    G.nodes[node]["tx_count"] = tx_count.get(node, 0)
    G.nodes[node]["color"] = f"#{random.randint(0, 0xFFFFFF):06x}"

# ==== Centrality ====
degree_centrality = nx.degree_centrality(G)
betweenness = nx.betweenness_centrality(G)

for node in G.nodes():
    G.nodes[node]["degree_centrality"] = degree_centrality.get(node, 0)
    G.nodes[node]["betweenness"] = betweenness.get(node, 0)

# ==== Output folder ====
output_folder = "results"
os.makedirs(output_folder, exist_ok=True)

# ==== CSV Summary ====
summary = []
for node, attr in G.nodes(data=True):
    summary.append({
        "Wallet": node,
        "Tx_Count": attr.get("tx_count", 0),
        "Degree_Centrality": attr.get("degree_centrality", 0),
        "Betweenness_Centrality": attr.get("betweenness", 0),
    })

summary_df = pd.DataFrame(summary).sort_values(by="Tx_Count", ascending=False)
csv_path = os.path.join(output_folder, "bitcoin_top100_wallets.csv")
summary_df.to_csv(csv_path, index=False)
print(f"✅ CSV summary saved to: {csv_path}")

# ==== Normalize node sizes ====
scaler_desktop = MinMaxScaler(feature_range=(100, 1000))
tx_counts = [[G.nodes[n]["tx_count"]] for n in G.nodes()]
scaled_sizes_desktop = scaler_desktop.fit_transform(tx_counts).flatten()

scaler_phone = MinMaxScaler(feature_range=(40, 300))
scaled_sizes_phone = scaler_phone.fit_transform(tx_counts).flatten()

# ==== Positions and labels ====
pos = nx.spring_layout(G, seed=42, k=0.5)
node_colors = [G.nodes[n]["color"] for n in G.nodes()]
labels_short = {n: n[:6] + "..." for n in G.nodes()}
labels_very_short = {n: n[:4] + "..." for n in G.nodes()}

# ==== Desktop graph ====
plt.figure(figsize=(18, 14))
nx.draw(G, pos, with_labels=False, node_size=scaled_sizes_desktop,
        node_color=node_colors, edge_color="gray", alpha=0.7, width=0.5)
nx.draw_networkx_labels(G, pos, labels_short, font_size=6)
plt.title("🔗 Bitcoin Token Transfers – Top 100 Wallets", fontsize=18)
img_path_desktop = os.path.join(output_folder, "bitcoin_wallet_network.png")
plt.savefig(img_path_desktop, dpi=300, bbox_inches="tight")
plt.show()
print(f"📸 Desktop graph saved to: {img_path_desktop}")


# ==== Phone graph ====
plt.figure(figsize=(7, 10))
nx.draw(G, pos, with_labels=False, node_size=scaled_sizes_phone,
        node_color=node_colors, edge_color="gray", alpha=0.7, width=0.3)
nx.draw_networkx_labels(G, pos, labels_very_short, font_size=5)
plt.title("🔗 Top 100 Bitcoin Wallets (Mobile)", fontsize=12)
img_path_phone = os.path.join(output_folder, "bitcoin_wallet_network_phone.png")
plt.savefig(img_path_phone, dpi=200, bbox_inches="tight")
plt.show()
print(f"📱 Mobile-friendly graph saved to: {img_path_phone}")
