import pandas as pd
import os

# === 1. Caricamento dati ===
net = pd.read_csv("../metrics/net_flow_from_export_only.csv", index_col=0)
btw = pd.read_csv("../metrics/betweenness.csv", index_col=0)

# === 2. Merge ===
merged = net.join(btw, how="inner")

# === 3. Ranking (facoltativo ma utile per visualizzazione) ===
merged["balance_rank"] = merged["balance_MWh"].rank(ascending=True)
merged["betweenness_rank"] = merged["betweenness"].rank(ascending=False)

# === 4. Salvataggio ===
os.makedirs("../metrics", exist_ok=True)
merged.to_csv("../metrics/merged_btw_net_2024.csv")

print("✅ File salvato in: ../metrics/merged_btw_net_2024.csv")

# === 5. Anteprima ===
print("\n🔍 Prime righe del file unificato:")
print(merged.head())
