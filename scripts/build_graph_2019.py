import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import os

# === 1. Caricamento dati ===
df = pd.read_csv("../data/processed/aggregated_flows_2019.csv")

# === 2. Costruzione grafo diretto ===
G = nx.DiGraph()
for _, row in df.iterrows():
    G.add_edge(row["from"], row["to"], weight=row["total_GWh"])

print(f"✔️ Grafo 2019 creato con {G.number_of_nodes()} nodi e {G.number_of_edges()} archi")

# === 3. Calcolo metriche ===
degree_in = dict(G.in_degree())
degree_out = dict(G.out_degree())
strength_in = dict(G.in_degree(weight='weight'))
strength_out = dict(G.out_degree(weight='weight'))
betweenness = nx.betweenness_centrality(G, weight='weight')

# === 4. Salvataggio metriche ===
os.makedirs("../metrics_2019", exist_ok=True)

pd.DataFrame.from_dict(degree_in, orient="index", columns=["in_degree"]).to_csv("../metrics_2019/in_degree.csv")
pd.DataFrame.from_dict(degree_out, orient="index", columns=["out_degree"]).to_csv("../metrics_2019/out_degree.csv")
pd.DataFrame.from_dict(strength_in, orient="index", columns=["in_strength_GWh"]).to_csv("../metrics_2019/in_strength.csv")
pd.DataFrame.from_dict(strength_out, orient="index", columns=["out_strength_GWh"]).to_csv("../metrics_2019/out_strength.csv")
pd.DataFrame.from_dict(betweenness, orient="index", columns=["betweenness"]).to_csv("../metrics_2019/betweenness.csv")

print("📁 Metriche 2019 salvate nella cartella 'metrics_2019/'.")

# === 5. Visualizzazione del grafo ===
plt.figure(figsize=(12, 10))
pos = nx.spring_layout(G, k=0.3, seed=42)

node_sizes = [strength_out.get(n, 0) / 100 for n in G.nodes]

nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color='lightgreen')
nx.draw_networkx_edges(G, pos, alpha=0.3)
nx.draw_networkx_labels(G, pos, font_size=8)

plt.title("🌍 European Electricity Trade Network (2019)", fontsize=14)
plt.axis("off")
plt.tight_layout()

# Salva nella cartella figures/
os.makedirs("../figures_2019", exist_ok=True)
plt.savefig("../figures_2019/network_map_2019.png")
plt.show()
