import pandas as pd
import os

# === 1. Caricamento dataset disaggregato ===
df = pd.read_csv("../data/physical_energy_and_power_flows", sep=";")

# === 2. Filtraggio solo righe Export ===
df = df[df["Direction"] == "Export"]
df = df[["FromAreaCode", "ToAreaCode", "Value"]].dropna()
df["Value"] = pd.to_numeric(df["Value"], errors="coerce")
df = df.dropna(subset=["Value"])

# === 3. Calcolo export e import basati solo su export dichiarato ===
exports = df.groupby("FromAreaCode")["Value"].sum().rename("export_MWh")
imports = df.groupby("ToAreaCode")["Value"].sum().rename("import_MWh")

# === 4. Unione e calcolo bilancio ===
net_flow = pd.concat([exports, imports], axis=1).fillna(0)
net_flow["balance_MWh"] = net_flow["export_MWh"] - net_flow["import_MWh"]

# === 5. Output ===
print("\n🔝 Top 5 esportatori netti (surplus):")
print(net_flow.sort_values("balance_MWh", ascending=False).head(5))

print("\n🔻 Top 5 importatori netti (deficit):")
print(net_flow.sort_values("balance_MWh", ascending=True).head(5))

# === 6. Salvataggio CSV ===
os.makedirs("../metrics", exist_ok=True)
net_flow.to_csv("../metrics/net_flow_from_export_only.csv")
print("\n📁 Dati salvati in: metrics/net_flow_from_export_only.csv")
