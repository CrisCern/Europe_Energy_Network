# ⚡ Europe Energy Network

**Network Analysis of Cross-Border Electricity Flows in Europe (2024)**  
Progetto per l’esame di *Economia dei Network*

---

## 🎯 Obiettivo

Analizzare la rete degli scambi di energia elettrica tra paesi europei, utilizzando dati reali del 2024 forniti da ENTSO-E.  
L’obiettivo è comprendere la struttura della rete, identificare i paesi più centrali, esplorare eventuali relazioni di dipendenza e valutare la flessibilità del sistema elettrico europeo.

---

## 🧠 Domande di ricerca

- **Chi sono i paesi più centrali nella rete elettrica europea?**
- **Quali paesi hanno la maggiore diversificazione nelle connessioni (in entrata e in uscita)?**
- **Si possono individuare comunità regionali o blocchi cooperativi stabili?**
- **Esistono segnali di squilibrio o eccessiva dipendenza all’interno della rete?**
- **Chi rappresenta un potenziale “collo di bottiglia” nella rete, ovvero un nodo la cui assenza comprometterebbe gravemente la circolazione dell’energia?**


---

## 📊 Approccio

L’analisi si basa su tecniche di **Network Analysis** applicate a grafi costruiti a partire dai flussi fisici mensili di energia elettrica.  
I dati sono elaborati per ottenere una visione ad alto livello delle dinamiche energetiche tra i paesi europei.

Sono previsti:
- Grafi **diretti e pesati** (per analizzare la direzione dei flussi)
- Eventuali grafi **non orientati** (per osservare la struttura complessiva delle connessioni)
- Calcolo di **metriche di centralità**
- Esplorazione di **comunità o cluster regionali**
- (Facoltativo) Rappresentazioni visive, analisi di flussi netti o evoluzioni temporali

---

## 📁 Struttura del progetto

La struttura del progetto è in evoluzione e verrà adattata sulla base delle esigenze analitiche.  
I file e gli script saranno organizzati in modo chiaro e progressivo, mantenendo coerenza tra codice, dati e documentazione.

---

## 🧰 Tecnologie e strumenti

- **Python 3.11+**
- `pandas`, `networkx`, `matplotlib`
- (eventuali: `plotly`, `pyvis`, `community`)

---

## 📎 Autori
Antonella Floris
Cristian Cernicchiaro  
Piera Marongiu  
Laurea Magistrale in Data Science, Business Analytics & Innovazione  
Università di Cagliari – A.A. 2024/2025


https://newtransparency.entsoe.eu/load/total/dayAhead?appState=%7B%22sa%22%3A%5B%22BZN%7C10Y1001A1001A71M%22%5D%2C%22st%22%3A%22BZN%22%2C%22mm%22%3Atrue%2C%22ma%22%3Afalse%2C%22sp%22%3A%22HALF%22%2C%22dt%22%3A%22TABLE%22%2C%22df%22%3A%5B%222025-05-26%22%2C%222025-05-26%22%5D%2C%22tz%22%3A%22CET%22%7D