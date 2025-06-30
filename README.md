
# ⚡ Europe Energy Network

**Network Analysis of Cross-Border Electricity Flows in Europe (2019 vs 2024)**  
*Project for the Network Economics course*

---

## 🎯 Objective

This project analyzes the network of cross-border electricity exchanges between European countries, using real data from **2019 and 2024** provided by ENTSO-E.  
The goal is to understand how the structure of the electricity network has evolved over time, identify the most central countries, explore dependency patterns, and assess the flexibility and resilience of the European energy system.

---

## 🧠 Research Questions

- Which countries are the most central in the European electricity network?  
- Which countries show the greatest diversification in inbound and outbound energy flows?  
- Are there identifiable regional communities or stable cooperative blocs?  
- Are there signs of imbalance or excessive dependency within the network?  
- Which countries act as potential bottlenecks — nodes whose removal would severely impact energy circulation?  
- How has the network evolved between 2019 and 2024?

---

## 📊 Methodology

The analysis is based on **network science techniques** applied to graphs constructed from monthly physical electricity flow data.  
Data is processed to provide a high-level, comparative overview of European energy dynamics over time.

Key steps include:
- Building **directed and weighted graphs** for both years  
- Comparative **centrality analysis** (degree, betweenness, eigenvector)  
- **Community detection** using the Louvain method  
- Calculation of **net energy balances** per country  
- Identification of **critical nodes and edges** (via betweenness centrality)  
- **Visual and analytical comparison** between the 2019 and 2024 networks

---

## 📈 Key Insights

- **Shift in centrality** towards Eastern and Northern Europe (e.g. Ukraine, Finland, Poland)
- **France and Sweden** emerged as major exporters; **Italy, UK, Germany** showed strong deficits
- New **critical corridors** in 2024: Russia–Ukraine and Georgia–Turkey
- **High dependency zones** identified in the Balkans and Caucasus
- The European grid shows increasing **polarization and complexity**, requiring long-term strategic attention

---

## 📁 Project Structure

data/ # Input datasets (2019 & 2024)
scripts/ # Preprocessing and network analysis scripts
figures/ # Generated plots and graphs
report/ # Final report and slides


---

## 🧰 Technologies & Tools

- **Python 3.11+**
- `pandas`, `networkx`, `matplotlib`  
- (optional: `plotly`, `pyvis`, `community`)

---

## 📄 Deliverables

- 📝 Final report with methodology, analysis, and results  
- 📊 Presentation slides  
- 🧾 Annotated and reusable Python scripts  

---

## 📎 Authors

**Antonella Floris**  
**Cristian Cernicchiaro**  
**Piera Marongiu**  

Master’s Degree in Data Science, Business Analytics & Innovation  
University of Cagliari – Academic Year 2024/2025
