import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# === 1. Caricamento dati ===
df = pd.read_csv("../metrics/centrality_metrics_2024.csv")

# Selezione dei top 10 paesi per Betweenness
top10 = df.sort_values(by="Betweenness", ascending=False).head(10)

# === 2. Grafico 1: Barplot Betweenness ===
sns.set(style="whitegrid")
plt.figure(figsize=(12, 6))
sns.barplot(data=top10, x="Paese", y="Betweenness", palette="viridis")
plt.title("Top 10 Paesi per Betweenness Centrality (2024)", fontsize=14)
plt.ylabel("Betweenness")
plt.xlabel("Paese")
plt.tight_layout()
plt.savefig("../figures/top10_betweenness.png")
plt.show()

# === 3. Grafico 2: Scatter Strength vs Betweenness ===
plt.figure(figsize=(10, 6))
sns.scatterplot(data=top10, x="Betweenness", y="Strength_GWh", hue="Paese", s=100)
plt.title("Betweenness vs Strength - Paesi più centrali (2024)", fontsize=14)
plt.xlabel("Betweenness Centrality")
plt.ylabel("Strength (GWh)")
plt.tight_layout()
plt.savefig("../figures/betweenness_vs_strength.png")
plt.show()

# === 4. Grafico 3: Radar Plot (Top 5 Paesi) ===
top5 = top10.head(5).set_index("Paese")
metrics = ["Degree", "Strength_GWh", "Betweenness"]
normalized = top5[metrics].copy()

# Normalizzazione 0-1 per confronto visivo
for col in metrics:
    normalized[col] = (normalized[col] - normalized[col].min()) / (normalized[col].max() - normalized[col].min())

# Radar chart setup
labels = metrics
num_vars = len(labels)
angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
angles += angles[:1]  # chiude il cerchio

# Plot radar
plt.figure(figsize=(8, 8))
for idx, row in normalized.iterrows():
    values = row.tolist() + row.tolist()[:1]
    plt.polar(angles, values, label=idx, linewidth=2)

plt.xticks(angles[:-1], labels, fontsize=12)
plt.title("Radar Plot – Profili di Centralità (Top 5 Paesi)", fontsize=14)
plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
plt.tight_layout()
plt.savefig("../figures/radar_top5.png")
plt.show()
