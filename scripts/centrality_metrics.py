import pandas as pd

# Caricamento e rinomina colonne
bet = pd.read_csv("../metrics/betweenness.csv").rename(columns={"Unnamed: 0": "country"})
deg = pd.read_csv("../metrics/in_degree.csv").rename(columns={"Unnamed: 0": "country"})
strg = pd.read_csv("../metrics/in_strength.csv").rename(columns={"Unnamed: 0": "country"})
part = pd.read_csv("../metrics/louvain_partition.csv")  # già corretto

# Merge su 'country'
df = bet.merge(deg, on='country') \
        .merge(strg, on='country') \
        .merge(part, on='country')

# Rinomina colonne finali
df.columns = ['Paese', 'Betweenness', 'Degree', 'Strength_GWh', 'Community']

# Ordina e salva
df = df.sort_values(by='Betweenness', ascending=False).reset_index(drop=True)
df.to_csv("../metrics/centrality_metrics_2024.csv", index=False)

print(" File centrality_metrics_2024.csv salvato correttamente.")
