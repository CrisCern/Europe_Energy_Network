import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import community.community_louvain as community_louvain  # Assicurati di avere installato: pip install python-louvain
import os

# === 1. Caricamento dati ===
df = pd.read_csv("../data/processed/aggregated_flows_2019.csv")

# === 2. Costruzione grafo non orientato ===
df_undirected = df.copy()
df_undirected["min_node"] = df_undirected[["from", "to"]].min(axis=1)
df_undirected["max_node"] = df_undirected[["from", "to"]].max(axis=1)

df_grouped = (
    df_undirected.groupby(["min_node", "max_node"])["total_GWh"]
    .sum()
    .reset_index()
    .rename(columns={"min_node": "node1", "max_node": "node2"})
)

G = nx.Graph()
for _, row in df_grouped.iterrows():
    G.add_edge(row["node1"], row["node2"], weight=row["total_GWh"])

print(f"✔️ Grafo NON orientato creato con {G.number_of_nodes()} nodi e {G.number_of_edges()} archi")

# === 3. Community detection con Louvain ===
partition = community_louvain.best_partition(G, weight='weight')

# === 4. Salvataggio comunità ===
os.makedirs("../metrics_2019", exist_ok=True)
df_partition = pd.DataFrame.from_dict(partition, orient='index', columns=['community'])
df_partition.index.name = 'country'
df_partition.to_csv("../metrics_2019/louvain_partition_2020.csv")

print("\n🔍 Comunità rilevate (prime 5 righe):")
print(df_partition.head())

# === 5. Visualizzazione del grafo con colori per comunità ===
plt.figure(figsize=(12, 10))
pos = nx.spring_layout(G, k=0.3, seed=42)

colors = [partition[node] for node in G.nodes()]
nx.draw_networkx_nodes(G, pos, node_color=colors, cmap=plt.cm.Set3, node_size=300)
nx.draw_networkx_edges(G, pos, alpha=0.3)
nx.draw_networkx_labels(G, pos, font_size=8)

plt.title("European Electricity Network – Louvain Communities (2020)", fontsize=14)
plt.axis("off")
plt.tight_layout()

os.makedirs("../figures_2019", exist_ok=True)
plt.savefig("../figures_2019/network_map_communities_2020.png")
plt.show()
