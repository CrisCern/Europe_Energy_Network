import pandas as pd
import os

# === 1. Caricamento dati ===
net = pd.read_csv("../metrics_2019/net_flow_from_export_only_2019.csv", index_col=0)
btw = pd.read_csv("../metrics_2019/betweenness.csv", index_col=0)

# === 2. Merge ===
merged = net.join(btw, how="inner")

# === 3. Calcolo ranking (facoltativo ma utile)
merged["balance_rank"] = merged["balance_MWh"].rank(ascending=True)
merged["betweenness_rank"] = merged["betweenness"].rank(ascending=False)

# === 4. Salvataggio ===
os.makedirs("../metrics_2019", exist_ok=True)
merged.to_csv("../metrics_2019/merged_btw_net_2019.csv")

print("✅ File salvato in: ../metrics_2019/merged_btw_net_2019.csv")

# === 5. Anteprima ===
print("\n🔍 Prime righe del file unificato:")
print(merged.head())
