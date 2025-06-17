import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import os
import community as community_louvain
from collections import Counter
import seaborn as sns
import numpy as np
from itertools import combinations
import math
from tqdm import tqdm

class EnergyDataLoader:
    def __init__(self, path, year=None, export_only=False, sep=";"):
        self.path = path
        self.year = year
        self.export_only = export_only
        self.sep = sep
        self.df = None

    def load_and_aggregate(self):
        df = pd.read_csv(self.path, sep=self.sep)

        if self.year:
            df = df[df["Year"] == self.year]

        if self.export_only:
            df = df[df["Direction"] == "Export"]

        df = df[["FromAreaCode", "ToAreaCode", "Value"]].dropna()
        df["Value"] = pd.to_numeric(df["Value"], errors="coerce")
        df = df.dropna(subset=["Value"])

        aggregated = (
            df.groupby(["FromAreaCode", "ToAreaCode"])["Value"]
            .sum()
            .reset_index()
            .rename(columns={"FromAreaCode": "from", "ToAreaCode": "to", "Value": "total_GWh"})
        )

        self.df = aggregated
        return aggregated

    def save(self, output_path):
        if self.df is not None:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            self.df.to_csv(output_path, index=False)
            print(f" File salvato: {output_path}")
        else:
            print(" Nessun dataframe da salvare.")


class EnergyGraphBuilder:
    def __init__(self, df, label):
        self.df = df
        self.label = label
        self.G = None
        self.metrics = {}

    def build_graph(self):
        G = nx.DiGraph()
        for _, row in self.df.iterrows():
            G.add_edge(row["from"], row["to"], weight=row["total_GWh"])
        self.G = G
        print(f" Grafo creato con {G.number_of_nodes()} nodi e {G.number_of_edges()} archi")
        return G

    def compute_metrics(self):
        self.metrics[f"in_degree_{self.label}"] = dict(self.G.in_degree())
        self.metrics[f"out_degree_{self.label}"] = dict(self.G.out_degree())
        self.metrics[f"in_strength_{self.label}"] = dict(self.G.in_degree(weight='weight'))
        self.metrics[f"out_strength_{self.label}"] = dict(self.G.out_degree(weight='weight'))
        self.metrics[f"betweenness_{self.label}"] = nx.betweenness_centrality(self.G, weight='weight')
        print(" Metriche calcolate.")
        return self.metrics

    def save_metrics(self, output_dir):
        os.makedirs(output_dir, exist_ok=True)
        for name, metric in self.metrics.items():
            df = pd.DataFrame(metric.items(), columns=["country", name])
            df.to_csv(os.path.join(output_dir, f"{name}.csv"), index=False)
        print(f" Metriche salvate in {output_dir}")

    def save_network_map(self, output_path):
        pos = nx.spring_layout(self.G, seed=42)
        node_sizes = [self.G.degree(n) * 50 for n in self.G.nodes()]
        plt.figure(figsize=(12, 10))
        nx.draw_networkx_nodes(self.G, pos, node_size=node_sizes, alpha=0.8, node_color='skyblue')
        nx.draw_networkx_edges(self.G, pos, alpha=0.3)
        nx.draw_networkx_labels(self.G, pos, font_size=8)
        plt.title(f"Network Map – {self.label}")
        plt.axis("off")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path)
        plt.close()
        print(f" Mappa salvata: {output_path}")

    def plot_degree_histograms(self, output_path):
        degree_in = dict(self.G.in_degree())
        degree_out = dict(self.G.out_degree())

        in_degree_values = list(degree_in.values())
        out_degree_values = list(degree_out.values())

        in_counts = Counter(in_degree_values)
        out_counts = Counter(out_degree_values)

        in_sorted = sorted(in_counts.items())
        out_sorted = sorted(out_counts.items())

        in_degrees, in_freqs = zip(*in_sorted)
        out_degrees, out_freqs = zip(*out_sorted)

        fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)

        axes[0].bar(in_degrees, in_freqs, color="dodgerblue")
        axes[0].set_title(f"In-degree Distribution – {self.label}")
        axes[0].set_xlabel("In-degree")
        axes[0].set_ylabel("Frequency")

        axes[1].bar(out_degrees, out_freqs, color="orange")
        axes[1].set_title(f"Out-degree Distribution – {self.label}")
        axes[1].set_xlabel("Out-degree")

        plt.tight_layout()
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path)
        plt.close()
        print(f" Istogramma gradi salvato: {output_path}")


class CommunityDetector:
    def __init__(self, df, label):
        self.df = df
        self.label = label
        self.G = None
        self.partition = None

    def build_undirected_graph(self):
        df_undirected = self.df.copy()
        df_undirected["min_node"] = df_undirected[["from", "to"]].min(axis=1)
        df_undirected["max_node"] = df_undirected[["from", "to"]].max(axis=1)

        df_grouped = (
            df_undirected.groupby(["min_node", "max_node"])["total_GWh"]
            .sum()
            .reset_index()
            .rename(columns={"min_node": "node1", "max_node": "node2"})
        )

        G = nx.Graph()
        for _, row in df_grouped.iterrows():
            G.add_edge(row["node1"], row["node2"], weight=row["total_GWh"])

        self.G = G
        print(f" Grafo NON orientato creato con {G.number_of_nodes()} nodi e {G.number_of_edges()} archi")

    def detect_communities(self):
        self.partition = community_louvain.best_partition(self.G, weight='weight')
        print(" Partizione Louvain calcolata.")
        return self.partition

    def save_partition(self, output_path):
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df_partition = pd.DataFrame.from_dict(self.partition, orient='index', columns=['community'])
        df_partition.index.name = 'country'
        df_partition.to_csv(output_path)
        print(f" Partizione salvata: {output_path}")

    def save_partition_map(self, output_path):
        pos = nx.spring_layout(self.G, seed=42)
        colors = [self.partition[node] for node in self.G.nodes()]

        plt.figure(figsize=(12, 10))
        nx.draw_networkx_nodes(self.G, pos, node_color=colors, cmap=plt.cm.Set3, node_size=300)
        nx.draw_networkx_edges(self.G, pos, alpha=0.3)
        nx.draw_networkx_labels(self.G, pos, font_size=8)
        plt.title(f"Community Map – {self.label}")
        plt.axis("off")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path)
        plt.close()
        print(f" Mappa con comunità salvata: {output_path}")


class CentralityMerger:
    def __init__(self, metrics_dir, label):
        self.metrics_dir = metrics_dir
        self.label = label

    def merge_and_save(self):
        bet = pd.read_csv(os.path.join(self.metrics_dir, f"betweenness_{self.label}.csv"))
        deg = pd.read_csv(os.path.join(self.metrics_dir, f"in_degree_{self.label}.csv"))
        strg = pd.read_csv(os.path.join(self.metrics_dir, f"in_strength_{self.label}.csv"))
        part = pd.read_csv(os.path.join(self.metrics_dir, f"louvain_partition_{self.label}.csv"))

        bet = bet.rename(columns={bet.columns[0]: "country"})
        deg = deg.rename(columns={deg.columns[0]: "country"})
        strg = strg.rename(columns={strg.columns[0]: "country"})

        df = bet.merge(deg, on='country')\
                .merge(strg, on='country')\
                .merge(part, on='country')

        df.columns = ['Paese', 'Betweenness', 'Degree', 'Strength_GWh', 'Community']
        df = df.sort_values(by='Betweenness', ascending=False).reset_index(drop=True)

        output_path = os.path.join(self.metrics_dir, f"centrality_metrics_{self.label}.csv")
        df.to_csv(output_path, index=False)
        print(f" File {output_path} salvato correttamente.")




class CentralityVisualizer:
    def __init__(self):
        pass

    def plot_top10_betweenness(self, csv_path, output_path):
        df = pd.read_csv(csv_path)
        top10 = df.sort_values(by="Betweenness", ascending=False).head(10)

        sns.set(style="whitegrid")
        plt.figure(figsize=(12, 6))
        sns.barplot(data=top10, x="Paese", y="Betweenness", palette="viridis")
        plt.title("Top 10 Paesi per Betweenness Centrality", fontsize=14)
        plt.ylabel("Betweenness")
        plt.xlabel("Paese")
        plt.tight_layout()
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path)
        plt.close()
        print(f" Barplot salvato: {output_path}")

    def plot_betweenness_vs_strength(self, csv_path, output_path):
        df = pd.read_csv(csv_path)
        top10 = df.sort_values(by="Betweenness", ascending=False).head(10)

        plt.figure(figsize=(10, 6))
        sns.scatterplot(data=top10, x="Betweenness", y="Strength_GWh", hue="Paese", s=100)
        plt.title("Betweenness vs Strength - Paesi più centrali", fontsize=14)
        plt.xlabel("Betweenness Centrality")
        plt.ylabel("Strength (GWh)")
        plt.tight_layout()
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path)
        plt.close()
        print(f" Scatter salvato: {output_path}")

    def plot_radar_top5(self, csv_path, output_path):
        df = pd.read_csv(csv_path)
        top5 = df.sort_values(by="Betweenness", ascending=False).head(5).set_index("Paese")
        metrics = ["Degree", "Strength_GWh", "Betweenness"]
        normalized = top5[metrics].copy()

        for col in metrics:
            normalized[col] = (normalized[col] - normalized[col].min()) / (
                        normalized[col].max() - normalized[col].min())

        labels = metrics
        angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
        angles += angles[:1]

        plt.figure(figsize=(8, 8))
        for idx, row in normalized.iterrows():
            values = row.tolist() + row.tolist()[:1]
            plt.polar(angles, values, label=idx, linewidth=2)

        plt.xticks(angles[:-1], labels, fontsize=12)
        plt.title("Radar Plot – Profili di Centralità (Top 5 Paesi)", fontsize=14)
        plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
        plt.tight_layout()
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path)
        plt.close()
        print(f" Radar plot salvato: {output_path}")

class NetFlowAnalyzer:
    def __init__(self, filepath, output_path):
        self.filepath = filepath
        self.output_path = output_path
        self.df = None
        self.net_flow = None

    def load_and_filter(self):
        self.df = pd.read_csv(self.filepath, sep=";")
        self.df = self.df[self.df["Direction"] == "Export"]
        self.df = self.df[["FromAreaCode", "ToAreaCode", "Value"]].dropna()
        self.df["Value"] = pd.to_numeric(self.df["Value"], errors="coerce")
        self.df = self.df.dropna(subset=["Value"])

    def compute_net_flow(self):
        exports = self.df.groupby("FromAreaCode")["Value"].sum().rename("export_MWh")
        imports = self.df.groupby("ToAreaCode")["Value"].sum().rename("import_MWh")
        self.net_flow = pd.concat([exports, imports], axis=1).fillna(0)
        self.net_flow["balance_MWh"] = self.net_flow["export_MWh"] - self.net_flow["import_MWh"]

    def save_to_csv(self):
        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
        self.net_flow.to_csv(self.output_path)
        print(f" Dati salvati in: {self.output_path}")

    def print_top_countries(self, label=""):
        print(f"\n Top 5 esportatori netti (surplus){label}:")
        print(self.net_flow.sort_values("balance_MWh", ascending=False).head(5))
        print(f"\n Top 5 importatori netti (deficit){label}:")
        print(self.net_flow.sort_values("balance_MWh", ascending=True).head(5))


class MergedMetricsBuilder:
    def __init__(self, net_path, btw_path, output_path):
        self.net_path = net_path
        self.btw_path = btw_path
        self.output_path = output_path
        self.merged = None

    def build(self):
        net = pd.read_csv(self.net_path, index_col=0)
        btw = pd.read_csv(self.btw_path, index_col=0)
        self.merged = net.join(btw, how="inner")

        # Gestione flessibile dei nomi colonne
        balance_col = next((col for col in self.merged.columns if "balance_MWh" in col), None)
        btw_col = next((col for col in self.merged.columns if "betweenness" in col.lower()), None)

        if not balance_col or not btw_col:
            raise ValueError(f" Colonne richieste non trovate. Colonne disponibili: {self.merged.columns.tolist()}")

        self.merged["balance_rank"] = self.merged[balance_col].rank(ascending=True)
        self.merged["betweenness_rank"] = self.merged[btw_col].rank(ascending=False)

    def save(self):
        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
        self.merged.to_csv(self.output_path)
        print(f" File salvato in: {self.output_path}")

    def preview(self, n=5):
        print(f"\n Prime {n} righe del file unificato:")
        print(self.merged.head(n))


class CriticalNodeComparer:
    def __init__(self, path_2019, path_2024, output_path):
        self.path_2019 = path_2019
        self.path_2024 = path_2024
        self.output_path = output_path
        self.crit_merged = None

    def compare(self):
        df_2019 = pd.read_csv(self.path_2019, index_col=0)
        df_2024 = pd.read_csv(self.path_2024, index_col=0)

        # NON serve rinominare, usiamo direttamente i nomi presenti
        df_2019 = df_2019[["balance_MWh", "betweenness_2019"]].rename(
            columns={"balance_MWh": "balance_MWh_2019"}
        )
        df_2024 = df_2024[["balance_MWh", "betweenness_2024"]].rename(
            columns={"balance_MWh": "balance_MWh_2024"}
        )

        merged = df_2019.join(df_2024, how="inner")

        crit_2019 = merged[
            (merged["balance_MWh_2019"] < 0) &
            (merged["betweenness_2019"] > merged["betweenness_2019"].mean())
            ]
        crit_2024 = merged[
            (merged["balance_MWh_2024"] < 0) &
            (merged["betweenness_2024"] > merged["betweenness_2024"].mean())
            ]

        crit_nodes = set(crit_2019.index).union(set(crit_2024.index))
        self.crit_merged = merged.loc[list(crit_nodes)]

    def plot_comparison(self):
        if self.crit_merged is None:
            raise ValueError("Devi prima eseguire compare().")

        x = range(len(self.crit_merged))
        width = 0.35

        plt.figure(figsize=(10, 6))
        plt.bar([i - width/2 for i in x], self.crit_merged["balance_MWh_2019"], width=width, label="2019")
        plt.bar([i + width/2 for i in x], self.crit_merged["balance_MWh_2024"], width=width, label="2024")
        plt.axhline(0, color="gray", linestyle="--", linewidth=0.7)
        plt.xticks(x, self.crit_merged.index, rotation=45)
        plt.title(" Paesi Critici: Net Balance 2019 vs 2024")
        plt.ylabel("Balance (MWh)")
        plt.legend()
        plt.tight_layout()
        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
        plt.savefig(self.output_path)
        plt.close()
        print(f" Grafico di confronto salvato: {self.output_path}")


def _plot_scatter_and_bar(self, df, year):
    # Rinomina la colonna betweenness_XXXX in betweenness (se necessario)
    btw_col = f"betweenness_{year}"
    if btw_col in df.columns:
        df = df.rename(columns={btw_col: "betweenness"})

    # Rinomina indice se serve
    if "country" in df.columns:
        df.set_index("country", inplace=True)

    # Scatterplot
    mean_btw = df["betweenness"].mean()
    plt.figure(figsize=(10, 6))
    plt.scatter(df["balance_MWh"], df["betweenness"], color="darkcyan", alpha=0.7)
    plt.axvline(0, color="red", linestyle="--", linewidth=1)
    plt.axhline(mean_btw, color="orange", linestyle="--", linewidth=1)

    for i, row in df.iterrows():
        if row["balance_MWh"] < 0 and row["betweenness"] > mean_btw:
            plt.text(row["balance_MWh"], row["betweenness"], i, fontsize=8, color="black")

    plt.title(f"Paesi europei: Betweenness vs Bilancio energetico ({year})")
    plt.xlabel("Bilancio energetico netto (MWh)")
    plt.ylabel("Betweenness Centrality")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"../figures/balance_vs_betweenness_{year}.png")
    plt.show()

    # Barplot
    critici = df[(df["balance_MWh"] < 0) & (df["betweenness"] > mean_btw)]
    critici_sorted = critici.sort_values("betweenness", ascending=False)

    plt.figure(figsize=(10, 6))
    plt.bar(critici_sorted.index, critici_sorted["betweenness"], color="crimson")
    plt.title(f"Paesi critici nel {year} (Alta betweenness e deficit energetico)")
    plt.xlabel("Paese")
    plt.ylabel("Betweenness Centrality")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f"../figures/critical_nodes_barplot_{year}.png")
    plt.show()

class CriticalPlotter:
    def __init__(self, path_2019, path_2024):
        self.path_2019 = path_2019
        self.path_2024 = path_2024

    def plot_insights_2019(self):
        df = pd.read_csv(self.path_2019, index_col=0)
        self._plot_scatter_and_bar(df, year=2019)

    def plot_insights_2024(self):
        df = pd.read_csv(self.path_2024, index_col=0)
        self._plot_scatter_and_bar(df, year=2024)

    def _plot_scatter_and_bar(self, df, year):
        # Rinomina la colonna betweenness_XXXX in 'betweenness' se necessario
        btw_col = f"betweenness_{year}"
        if btw_col in df.columns:
            df = df.rename(columns={btw_col: "betweenness"})

        # Imposta l'indice sul paese se non già fatto
        if "country" in df.columns:
            df.set_index("country", inplace=True)

        # Scatterplot
        mean_btw = df["betweenness"].mean()
        plt.figure(figsize=(10, 6))
        plt.scatter(df["balance_MWh"], df["betweenness"], color="darkcyan", alpha=0.7)
        plt.axvline(0, color="red", linestyle="--", linewidth=1)
        plt.axhline(mean_btw, color="orange", linestyle="--", linewidth=1)

        for i, row in df.iterrows():
            if row["balance_MWh"] < 0 and row["betweenness"] > mean_btw:
                plt.text(row["balance_MWh"], row["betweenness"], i, fontsize=8, color="black")

        plt.title(f"Paesi europei: Betweenness vs Bilancio energetico ({year})")
        plt.xlabel("Bilancio energetico netto (MWh)")
        plt.ylabel("Betweenness Centrality")
        plt.grid(True)
        plt.tight_layout()
        os.makedirs("../figures", exist_ok=True)
        plt.savefig(f"../figures/balance_vs_betweenness_{year}.png")
        plt.close()

        # Barplot: solo paesi critici
        critici = df[(df["balance_MWh"] < 0) & (df["betweenness"] > mean_btw)]
        critici_sorted = critici.sort_values("betweenness", ascending=False)

        plt.figure(figsize=(10, 6))
        plt.bar(critici_sorted.index, critici_sorted["betweenness"], color="crimson")
        plt.title(f"Paesi critici nel {year} (Alta betweenness e deficit energetico)")
        plt.xlabel("Paese")
        plt.ylabel("Betweenness Centrality")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f"../figures/critical_nodes_barplot_{year}.png")
        plt.close()

class BalanceComparisonPlotter:
    def __init__(self, path_2019, path_2024, output_dir="../figures_comparison"):
        self.path_2019 = path_2019
        self.path_2024 = path_2024
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self.merged = self._load_and_merge()

    def _load_and_merge(self):
        df_2019 = pd.read_csv(self.path_2019, index_col=0)
        df_2024 = pd.read_csv(self.path_2024, index_col=0)

        # Verifica colonne richieste per ciascun file
        required_2019 = ["balance_MWh", "betweenness_2019"]
        required_2024 = ["balance_MWh", "betweenness_2024"]

        for col in required_2019:
            if col not in df_2019.columns:
                raise ValueError(f"Colonna '{col}' mancante nel file 2019.")

        for col in required_2024:
            if col not in df_2024.columns:
                raise ValueError(f"Colonna '{col}' mancante nel file 2024.")

        # Rinomina e seleziona
        df_2019 = df_2019[["balance_MWh", "betweenness_2019"]].rename(
            columns={"balance_MWh": "balance_MWh_2019"}
        )
        df_2024 = df_2024[["balance_MWh", "betweenness_2024"]].rename(
            columns={"balance_MWh": "balance_MWh_2024"}
        )

        return df_2019.join(df_2024, how="inner")

    def plot_line_comparison(self):
        plt.figure(figsize=(10, 6))
        for country in self.merged.index:
            delta = abs(self.merged.loc[country, "balance_MWh_2024"] - self.merged.loc[country, "balance_MWh_2019"])
            label = country if delta > 20000 else ""
            plt.plot(["2019", "2024"],
                     [self.merged.loc[country, "balance_MWh_2019"], self.merged.loc[country, "balance_MWh_2024"]],
                     marker="o", label=label, alpha=0.6)

        plt.axhline(0, color="gray", linestyle="--", linewidth=0.7)
        plt.title(" Variazione Net Energy Balance (2019 vs 2024)")
        plt.ylabel("Balance (MWh)")
        plt.grid(True)
        plt.legend(loc="best", fontsize=8)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "net_balance_comparison_lines.png"))
        plt.show()

    def plot_heatmap_comparison(self):
        balance_df = pd.DataFrame({
            "2019": self.merged["balance_MWh_2019"],
            "2024": self.merged["balance_MWh_2024"]
        }).sort_values("2024", ascending=False)

        plt.figure(figsize=(8, 12))
        sns.heatmap(balance_df, cmap="coolwarm", annot=True, fmt=".0f",
                    linewidths=0.5, cbar_kws={"label": "Net Energy Balance (MWh)"})
        plt.title(" Net Energy Balance per Country (2019 vs 2024)", fontsize=14)
        plt.xlabel("Year")
        plt.ylabel("Country")
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "heatmap_net_balance_2019_2024.png"))
        plt.show()

class ShapleyAnalyzer:
    def __init__(self, netflow_path, betweenness_path, output_csv, max_coalition_size=3, seed=42):
        self.netflow_path = netflow_path
        self.betweenness_path = betweenness_path
        self.output_csv = output_csv
        self.max_coalition_size = max_coalition_size
        self.seed = seed

    def _load_data(self):
        df_net = pd.read_csv(self.netflow_path, index_col=0)
        df_btw = pd.read_csv(self.betweenness_path, index_col=0)

        # Verifica colonne
        if "balance_MWh" not in df_net.columns:
            raise ValueError("Colonna 'balance_MWh' mancante nel file netflow.")

        # Adattiamo il nome se serve
        if "betweenness_2024" in df_btw.columns:
            df_btw = df_btw.rename(columns={"betweenness_2024": "betweenness"})
        elif "betweenness" not in df_btw.columns:
            raise ValueError("Colonna 'betweenness' mancante nel file di centralità.")

        # Unione dei due dataframe
        df = df_net[["balance_MWh"]].join(df_btw[["betweenness"]], how="inner")
        return df

    def _characteristic_function(self, coalition, df):
        if not coalition:
            return 0
        sub_df = df.loc[coalition]
        return sub_df["balance_MWh"].sum() * sub_df["betweenness"].mean()

    def compute_shapley_values(self):
        df = self._load_data()
        countries = list(df.index)
        n = len(countries)
        shapley_values = {country: 0 for country in countries}

        for country in tqdm(countries, desc="Calcolo Shapley Values"):
            others = [c for c in countries if c != country]
            marginal_contributions = []

            for k in range(self.max_coalition_size + 1):
                for subset in combinations(others, k):
                    subset = list(subset)
                    v_without = self._characteristic_function(subset, df)
                    v_with = self._characteristic_function(subset + [country], df)
                    weight = (math.factorial(len(subset)) * math.factorial(n - len(subset) - 1)) / math.factorial(n)
                    marginal_contributions.append(weight * (v_with - v_without))

            shapley_values[country] = sum(marginal_contributions)

        df_out = pd.DataFrame({
            "country": list(shapley_values.keys()),
            "shapley_value": list(shapley_values.values())
        }).sort_values("shapley_value", ascending=False)

        os.makedirs(os.path.dirname(self.output_csv), exist_ok=True)
        df_out.to_csv(self.output_csv, index=False)
        print(f" Shapley values salvati in: {self.output_csv}")

    def plot_shapley_values(self):
        if not os.path.exists(self.output_csv):
            raise FileNotFoundError(f"Il file {self.output_csv} non esiste.")

        df = pd.read_csv(self.output_csv).sort_values("shapley_value", ascending=False)

        plt.figure(figsize=(10, 6))
        bars = plt.barh(df["country"], df["shapley_value"], color="skyblue")
        plt.axvline(0, color='black', linewidth=0.8)
        plt.xlabel("Shapley Value")
        plt.title("Shapley Values per Paese")
        plt.gca().invert_yaxis()
        plt.tight_layout()

        output_path = os.path.join(os.path.dirname(self.output_csv), "shapley_values_plot.png")
        plt.savefig(output_path)
        print(f" Grafico salvato in: {output_path}")
        plt.show()


if __name__ == "__main__":
    # === 1. 2019 ===
    loader_2019 = EnergyDataLoader(
        "../data/raw/physical_energy_power_flows_2019.csv", export_only=True
    )
    df_2019 = loader_2019.load_and_aggregate()
    loader_2019.save("../data/processed/aggregated_flows_2019.csv")

    builder_2019 = EnergyGraphBuilder(df_2019, label="2019")
    G_2019 = builder_2019.build_graph()
    metrics_2019 = builder_2019.compute_metrics()
    builder_2019.save_metrics("../metrics_2019")
    builder_2019.save_network_map("../figures/network_map_2019.png")
    builder_2019.plot_degree_histograms("../figures/degree_distribution_2019.png")

    detector_2019 = CommunityDetector(df_2019, label="2019")
    detector_2019.build_undirected_graph()
    detector_2019.detect_communities()
    detector_2019.save_partition("../metrics_2019/louvain_partition_2019.csv")
    detector_2019.save_partition_map("../figures/network_map_communities_2019.png")

    merger_2019 = CentralityMerger("../metrics_2019", label="2019")
    merger_2019.merge_and_save()

    # === 2. 2024 ===
    loader_2024 = EnergyDataLoader(
        "../data/raw/physical_energy_and_power_flows.csv", year=2024
    )
    df_2024 = loader_2024.load_and_aggregate()
    loader_2024.save("../data/processed/aggregated_flows_2024.csv")

    builder_2024 = EnergyGraphBuilder(df_2024, label="2024")
    G_2024 = builder_2024.build_graph()
    metrics_2024 = builder_2024.compute_metrics()
    builder_2024.save_metrics("../metrics_2024")
    builder_2024.save_network_map("../figures/network_map_2024.png")
    builder_2024.plot_degree_histograms("../figures/degree_distribution_2024.png")

    detector_2024 = CommunityDetector(df_2024, label="2024")
    detector_2024.build_undirected_graph()
    detector_2024.detect_communities()
    detector_2024.save_partition("../metrics_2024/louvain_partition_2024.csv")
    detector_2024.save_partition_map("../figures/network_map_communities_2024.png")

    merger_2024 = CentralityMerger("../metrics_2024", label="2024")
    merger_2024.merge_and_save()

    visualizer = CentralityVisualizer()

    # === Plot centralità 2019 ===
    visualizer.plot_top10_betweenness(
        "../metrics_2019/centrality_metrics_2019.csv",
        "../figures/top10_betweenness_2019.png"
    )
    visualizer.plot_betweenness_vs_strength(
        "../metrics_2019/centrality_metrics_2019.csv",
        "../figures/betweenness_vs_strength_2019.png"
    )
    visualizer.plot_radar_top5(
        "../metrics_2019/centrality_metrics_2019.csv",
        "../figures/radar_top5_2019.png"
    )

    # === Plot centralità 2024 ===
    visualizer.plot_top10_betweenness(
        "../metrics_2024/centrality_metrics_2024.csv",
        "../figures/top10_betweenness_2024.png"
    )
    visualizer.plot_betweenness_vs_strength(
        "../metrics_2024/centrality_metrics_2024.csv",
        "../figures/betweenness_vs_strength_2024.png"
    )
    visualizer.plot_radar_top5(
        "../metrics_2024/centrality_metrics_2024.csv",
        "../figures/radar_top5_2024.png"
    )

    # === Merge metrica: Net Flow + Betweenness ===
    builder_2019 = MergedMetricsBuilder(
        "../metrics_2019/net_flow_from_export_only_2019.csv",
        "../metrics_2019/betweenness_2019.csv",
        "../metrics_2019/merged_btw_net_2019.csv"
    )
    builder_2019.build()
    builder_2019.save()
    builder_2019.preview()

    builder_2024 = MergedMetricsBuilder(
        "../metrics_2024/net_flow_from_export_only_2024.csv",
        "../metrics_2024/betweenness_2024.csv",
        "../metrics_2024/merged_btw_net_2024.csv"
    )
    builder_2024.build()
    builder_2024.save()
    builder_2024.preview()

    # === Analisi netta dei flussi 2019 ===
    analyzer_2019 = NetFlowAnalyzer(
        "../data/raw/physical_energy_power_flows_2019.csv",
        "../metrics_2019/net_flow_from_export_only_2019.csv"
    )
    analyzer_2019.load_and_filter()
    analyzer_2019.compute_net_flow()
    analyzer_2019.save_to_csv()
    analyzer_2019.print_top_countries(label=" - 2019")

    # === Analisi netta dei flussi 2024 ===
    analyzer_2024 = NetFlowAnalyzer(
        "../data/raw/physical_energy_and_power_flows.csv",
        "../metrics_2024/net_flow_from_export_only_2024.csv"
    )
    analyzer_2024.load_and_filter()
    analyzer_2024.compute_net_flow()
    analyzer_2024.save_to_csv()
    analyzer_2024.print_top_countries(label=" - 2024")

    # === Confronto finale tra 2019 e 2024: Paesi critici ===
    comparer = CriticalNodeComparer(
        "../metrics_2019/merged_btw_net_2019.csv",
        "../metrics_2024/merged_btw_net_2024.csv",
        "../figures/critical_nodes_comparison.png"
    )

    comparer.compare()
    comparer.plot_comparison()

    plotter = CriticalPlotter(
        path_2019="../metrics_2019/merged_btw_net_2019.csv",
        path_2024="../metrics_2024/merged_btw_net_2024.csv"
    )

    plotter.plot_insights_2019()
    plotter.plot_insights_2024()

    balance_plotter = BalanceComparisonPlotter(
        path_2019="../metrics_2019/merged_btw_net_2019.csv",
        path_2024="../metrics_2024/merged_btw_net_2024.csv",
        output_dir="../figures_comparison"
    )

    balance_plotter.plot_line_comparison()
    balance_plotter.plot_heatmap_comparison()



    # === Analisi cooperativa: Shapley Value (2024) ===
    shapley = ShapleyAnalyzer(
        netflow_path="../metrics_2024/net_flow_from_export_only_2024.csv",
        betweenness_path="../metrics_2024/merged_btw_net_2024.csv",
        output_csv="../metrics/shapley_values_2024.csv"
    )
    shapley.compute_shapley_values()
    shapley.plot_shapley_values()
