import pandas as pd
import matplotlib.pyplot as plt
import os

# === 1. Caricamento dati ===
df = pd.read_csv("../metrics/merged_btw_net_2024.csv", index_col=0)

# === 2. Scatterplot: Balance vs Betweenness ===
plt.figure(figsize=(10, 6))
plt.scatter(df["balance_MWh"], df["betweenness"], color="darkcyan", alpha=0.7)
plt.axvline(0, color="red", linestyle="--", linewidth=1)
plt.axhline(df["betweenness"].mean(), color="orange", linestyle="--", linewidth=1)

for i, row in df.iterrows():
    if row["balance_MWh"] < 0 and row["betweenness"] > df["betweenness"].mean():
        plt.text(row["balance_MWh"], row["betweenness"], i, fontsize=8, color="black")

plt.title("Paesi europei: Betweenness vs Bilancio energetico (2024)")
plt.xlabel("Bilancio energetico netto (MWh)")
plt.ylabel("Betweenness Centrality")
plt.grid(True)
plt.tight_layout()
os.makedirs("../figures", exist_ok=True)
plt.savefig("../figures/balance_vs_betweenness_2024.png")
plt.show()

# === 3. Barplot: Paesi critici ===
critici = df[(df["balance_MWh"] < 0) & (df["betweenness"] > df["betweenness"].mean())]
critici_sorted = critici.sort_values("betweenness", ascending=False)

plt.figure(figsize=(10, 6))
plt.bar(critici_sorted.index, critici_sorted["betweenness"], color="crimson")
plt.title("Paesi critici nel 2024 (Alta betweenness e deficit energetico)")
plt.xlabel("Paese")
plt.ylabel("Betweenness Centrality")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("../figures/critical_nodes_barplot_2024.png")
plt.show()
