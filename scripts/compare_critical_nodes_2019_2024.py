import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# === 1. Caricamento file 2019 e 2024 ===
df_2019 = pd.read_csv("../metrics_2019/merged_btw_net_2019.csv", index_col=0)
df_2024 = pd.read_csv("../metrics/merged_btw_net_2024.csv", index_col=0)

# === 2. Selezione colonne e rinomina ===
cols = ["balance_MWh", "betweenness"]
df_2019 = df_2019[cols].rename(columns=lambda c: f"{c}_2019")
df_2024 = df_2024[cols].rename(columns=lambda c: f"{c}_2024")

# === 3. Merge per confronto ===
merged = df_2019.join(df_2024, how="inner")

# === 4. Selezione nodi critici ===
crit_2019 = merged[(merged["balance_MWh_2019"] < 0) & (merged["betweenness_2019"] > merged["betweenness_2019"].mean())]
crit_2024 = merged[(merged["balance_MWh_2024"] < 0) & (merged["betweenness_2024"] > merged["betweenness_2024"].mean())]
crit_nodes = set(crit_2019.index).union(set(crit_2024.index))
crit_merged = merged.loc[list(crit_nodes)]

# === 5. Plot Bar Plot: Net Balance 2019 vs 2024 ===
x = range(len(crit_merged))
width = 0.35

plt.figure(figsize=(10, 6))
plt.bar([i - width/2 for i in x], crit_merged["balance_MWh_2019"], width=width, label="2019")
plt.bar([i + width/2 for i in x], crit_merged["balance_MWh_2024"], width=width, label="2024")
plt.axhline(0, color="gray", linestyle="--", linewidth=0.7)
plt.xticks(x, crit_merged.index, rotation=45)
plt.title("⚠️ Paesi Critici: Net Balance 2019 vs 2024")
plt.ylabel("Balance (MWh)")
plt.legend()
plt.tight_layout()
os.makedirs("../figures_comparison", exist_ok=True)
plt.savefig("../figures_comparison/critical_nodes_comparison.png")
plt.close()

# === 6. Heatmap variazioni normalizzate ===
heat_df = crit_merged[[
    "balance_MWh_2019", "balance_MWh_2024",
    "betweenness_2019", "betweenness_2024"
]].copy()

# Normalizzazione per confronto visivo
heat_df_norm = (heat_df - heat_df.min()) / (heat_df.max() - heat_df.min())

plt.figure(figsize=(12, 8))
sns.heatmap(heat_df_norm, cmap="coolwarm", annot=True, fmt=".2f", linewidths=.5, cbar_kws={"label": "Valore Normalizzato"})
plt.title("🔍 Heatmap - Paesi Critici: Balance e Betweenness (2019 vs 2024)")
plt.ylabel("Paesi")
plt.tight_layout()
plt.savefig("../figures_comparison/critical_nodes_heatmap.png")
plt.show()
