# Europe_Energy_Network
Progetto per Economia dei Network

Domanda di ricerca:
Who are the most central countries in the European electricity trade network,
and how are dependencies structured across the region?


-Struttura del dataset
Ogni riga rappresenta un flusso energetico mensile tra due Paesi. Le colonne chiave sono:
FromAreaCode: codice del Paese esportatore
ToAreaCode: codice del Paese importatore
Value: energia trasferita (in GWh)
Year, Month: data del flusso
Direction: Export o Import (può essere utile per filtrare)
MeasureItem: tutti i dati riguardano “Physical Energy & Power Flows”
Flag: indica se il dato è armonizzato
FromAreaMemberType, ToAreaMemberType: tipo di membro (es. "M" = Member)

-Come usarlo per il progetto
Sommiamo i Value mese per mese per ottenere un flusso annuale totale tra ogni coppia di paesi.
Costruiamo un grafo diretto e pesato:
Nodi = Paesi
Archi = flussi annuali (peso = GWh)

Analizziamo:
Centralità (degree, betweenness)
Chi esporta/chi importa di più
Cluster di cooperazione
Visualizziamo con NetworkX o Plotly.

-Struttura del Report

Introduction
Domanda di ricerca
Motivazione (contesto geopolitico, transizione energetica, dipendenze europee)
Data & Methodology
Origine e struttura del dataset
Pre-processing e costruzione del grafo
Spiegazione delle metriche di rete
Results
Paesi più centrali/importanti
Community di cooperazione
Visualizzazioni e insight
Conclusion
Sintesi dei risultati
Limiti del dataset
Proposte per ricerche future (es. integrazione con gas o PIL)
References
Link al dataset
Eventuali articoli/risorse citate