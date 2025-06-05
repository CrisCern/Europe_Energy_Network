import pandas as pd

# === Percorsi ===
RAW_DATA = "../data/raw/physical_energy_power_flows_2019.csv"
OUTPUT = "../data/processed/aggregated_flows_2019.csv"

# 1. Caricamento
df = pd.read_csv(RAW_DATA, sep=";")

# 2. Filtro solo Export
df = df[df["Direction"] == "Export"]

# 3. Pulizia colonne
df = df[["FromAreaCode", "ToAreaCode", "Value"]].dropna()
df["Value"] = pd.to_numeric(df["Value"], errors="coerce")
df = df.dropna(subset=["Value"])

# 4. Aggregazione
aggregated = (
    df.groupby(["FromAreaCode", "ToAreaCode"])["Value"]
    .sum()
    .reset_index()
    .rename(columns={"FromAreaCode": "from", "ToAreaCode": "to", "Value": "total_GWh"})
)

# 5. Salvataggio
aggregated.to_csv(OUTPUT, index=False)
print("✔️ Dati aggregati 2019 salvati in:", OUTPUT)
