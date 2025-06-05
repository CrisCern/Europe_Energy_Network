import pandas as pd
import os

# === 1. Caricamento dataset 2020 ===
df = pd.read_csv("../data/raw/physical_energy_power_flows_2019.csv", sep=";")

# === 2. Filtro solo Export ===
df = df[df["Direction"] == "Export"]
df = df[["FromAreaCode", "ToAreaCode", "Value"]].dropna()
df["Value"] = pd.to_numeric(df["Value"], errors="coerce")
df = df.dropna(subset=["Value"])

# === 3. Calcolo Export e Import basati solo su export ===
exports = df.groupby("FromAreaCode")["Value"].sum().rename("export_MWh")
imports = df.groupby("ToAreaCode")["Value"].sum().rename("import_MWh")

# === 4. Bilancio Netto ===
net_flow = pd.concat([exports, imports], axis=1).fillna(0)
net_flow["balance_MWh"] = net_flow["export_MWh"] - net_flow["import_MWh"]

# === 5. Salvataggio ===
os.makedirs("../metrics_2019", exist_ok=True)
net_flow.to_csv("../metrics_2019/net_flow_from_export_only_2020.csv")

# === 6. Output a schermo ===
print("\n🔝 Top 5 esportatori netti (surplus) - 2020:")
print(net_flow.sort_values("balance_MWh", ascending=False).head(5))

print("\n🔻 Top 5 importatori netti (deficit) - 2020:")
print(net_flow.sort_values("balance_MWh", ascending=True).head(5))
