import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# === 1. Caricamento file 2019 e 2024 ===
df_2019 = pd.read_csv("../metrics_2019/merged_btw_net_2019.csv", index_col=0)
df_2024 = pd.read_csv("../metrics/merged_btw_net_2024.csv", index_col=0)

# === 2. Selezione colonne rilevanti ===
cols = ["balance_MWh", "betweenness"]
df_2019 = df_2019[cols].rename(columns=lambda c: f"{c}_2019")
df_2024 = df_2024[cols].rename(columns=lambda c: f"{c}_2024")

# === 3. Merge per confronto ===
merged = df_2019.join(df_2024, how="inner")

# === 4. Line Plot variazione bilancio netto ===
plt.figure(figsize=(10, 6))
for country in merged.index:
    plt.plot(
        ["2019", "2024"],
        [merged.loc[country, "balance_MWh_2019"], merged.loc[country, "balance_MWh_2024"]],
        marker="o",
        label=country if abs(merged.loc[country, "balance_MWh_2024"] - merged.loc[country, "balance_MWh_2019"]) > 20000 else "",
        alpha=0.6
    )
plt.axhline(0, color="gray", linestyle="--", linewidth=0.7)
plt.title("📉 Variazione Net Energy Balance (2019 vs 2024)")
plt.ylabel("Balance (MWh)")
plt.grid(True)
plt.legend(loc="best", fontsize=8)
os.makedirs("../figures_comparison", exist_ok=True)
plt.tight_layout()
plt.savefig("../figures_comparison/net_balance_comparison_lines.png")
plt.show()

# === 5. Heatmap Net Energy Balance 2019 vs 2024 ===
balance_df = pd.DataFrame({
    "2019": merged["balance_MWh_2019"],
    "2024": merged["balance_MWh_2024"]
}).sort_values("2024", ascending=False)

plt.figure(figsize=(8, 12))
sns.heatmap(balance_df, cmap="coolwarm", annot=True, fmt=".0f", linewidths=0.5, cbar_kws={"label": "Net Energy Balance (MWh)"})
plt.title("🌍 Net Energy Balance per Country (2019 vs 2024)", fontsize=14)
plt.xlabel("Year")
plt.ylabel("Country")
plt.tight_layout()
plt.savefig("../figures_comparison/heatmap_net_balance_2019_2024.png")
plt.show()
