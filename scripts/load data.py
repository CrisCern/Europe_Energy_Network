import pandas as pd

# === Percorsi ===
RAW_DATA = "physical_energy_and_power_flows"
OUTPUT = "aggregated_flows_2024.csv"

# 1. Caricamento
df = pd.read_csv(RAW_DATA, sep=";")

# 2. Filtraggio 2024
df_2024 = df[df["Year"] == 2024]

# 3. Aggregazione
aggregated = (
    df_2024.groupby(["FromAreaCode", "ToAreaCode"])["Value"]
    .sum()
    .reset_index()
    .rename(columns={"FromAreaCode": "from", "ToAreaCode": "to", "Value": "total_GWh"})
)

# 4. Salvataggio
aggregated.to_csv(OUTPUT, index=False)
print("✔️ Dati aggregati salvati in:", OUTPUT)
