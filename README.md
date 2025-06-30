⚡ Europe Energy Network

Network Analysis of Cross-Border Electricity Flows in Europe (2019 vs 2024)
Project for the Network Economics course

🎯 Objective
This project analyzes the network of cross-border electricity exchanges between European countries, using real data from 2019 and 2024 provided by ENTSO-E.
The goal is to understand how the structure of the electricity network has evolved over time, identify the most central countries, explore dependency patterns, and assess the flexibility and resilience of the European energy system.

🧠 Research Questions
Which countries are the most central in the European electricity network?
Which countries show the greatest diversification in inbound and outbound energy flows?
Are there identifiable regional communities or stable cooperative blocs?
Are there signs of imbalance or excessive dependency within the network?
Which countries act as potential bottlenecks — nodes whose removal would severely impact energy circulation?
How has the network evolved between 2019 and 2024?

📊 Methodology
The analysis is based on network science techniques applied to graphs constructed from monthly physical electricity flow data.
Data is processed to provide a high-level, comparative overview of European energy dynamics over time.
Steps include:
Building directed and weighted graphs for both years
Comparative centrality analysis (degree, betweenness, eigenvector)
Community detection using the Louvain method
Calculation of net energy balances per country
Identification of critical nodes and critical edges using betweenness centrality
Visual and analytical comparison between 2019 and 2024 network structures

📈 Key Insights
Shift in network centrality towards Eastern and Northern Europe: Countries such as Ukraine, Finland, and Poland increased their strategic importance in 2024.
France and Sweden emerged as major exporters, while Italy, the UK, and Germany showed notable energy deficits.
Critical corridors evolved: in 2024, connections like Russia–Ukraine and Georgia–Turkey became central to the network.
High dependency zones were detected in the Balkans and Caucasus regions, with some countries highly reliant on a few neighbors.
The European grid shows increased complexity and polarization, requiring strategic attention for long-term energy security.
📁 Project Structure
The repository is organized into the following folders:
data/: input datasets (2019 & 2024)
scripts/: preprocessing and network analysis scripts
figures/: generated plots and graphs
report/: final report and slides

🧰 Technologies & Tools
Python 3.11+
pandas, networkx, matplotlib
(optional: plotly, pyvis, community)


📎 Authors
Antonella Floris
Cristian Cernicchiaro
Piera Marongiu
Master’s Degree in Data Science, Business Analytics & Innovation
University of Cagliari – Academic Year 2024/2025
