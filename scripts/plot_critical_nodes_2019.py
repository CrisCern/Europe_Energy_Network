import pandas as pd
import matplotlib.pyplot as plt
import os

# === 1. Caricamento dati ===
df = pd.read_csv("../metrics_2019/merged_btw_net_2019.csv", index_col=0)
if "country" in df.columns:
    df.set_index("country", inplace=True)

# === 2. Identificazione paesi critici ===
mean_btw = df["betweenness"].mean()
critici = df[(df["balance_MWh"] < 0) & (df["betweenness"] > mean_btw)]

# === 3. Scatterplot ===
plt.figure(figsize=(10, 6))
plt.scatter(df["balance_MWh"], df["betweenness"], color="darkcyan", alpha=0.7)
plt.axvline(0, color="red", linestyle="--", linewidth=1)
plt.axhline(mean_btw, color="orange", linestyle="--", linewidth=1)

for i, row in df.iterrows():
    if row["balance_MWh"] < 0 and row["betweenness"] > mean_btw:
        plt.text(row["balance_MWh"], row["betweenness"], i, fontsize=8, color="black")

plt.title("Paesi europei: Betweenness vs Bilancio energetico (2019)")
plt.xlabel("Bilancio energetico netto (MWh)")
plt.ylabel("Betweenness Centrality")
plt.grid(True)
plt.tight_layout()
os.makedirs("../figures_2019", exist_ok=True)
plt.savefig("../figures_2019/balance_vs_betweenness_2019.png")
plt.show()

# === 4. Barplot ===
critici_sorted = critici.sort_values("betweenness", ascending=False)

plt.figure(figsize=(10, 6))
plt.bar(critici_sorted.index, critici_sorted["betweenness"], color="crimson")
plt.title("Paesi critici nel 2019 (Alta betweenness e deficit energetico)")
plt.xlabel("Paese")
plt.ylabel("Betweenness Centrality")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("../figures_2019/critical_nodes_barplot_2019.png")
plt.show()
