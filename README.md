# Twitter Community Detection: Infomap vs. Leiden

**Team:** Jevon Lipsey, Zahara Love, Marcos Arnold

**Course:** Data Structures & Algorithms Final Project

<img width="1084" height="773" alt="example_cluster" src="https://github.com/user-attachments/assets/5510d6a3-8a64-4a0b-94c4-e8ee9f38fdbd" />


## Overview

This project implements and compares two advanced community detection algorithms—**Infomap** and **Leiden**—from scratch in Python. The goal was to analyze complex social network structures within the Stanford SNAP Twitter dataset, compare the algorithmic approaches (flow compression vs. modularity optimization), and identify political echo chambers.

## Key Features

* **From-Scratch Infomap:** A Python implementation of the Infomap algorithm based on the Map Equation and information flow compression.
* **From-Scratch Leiden:** A Python implementation of the Leiden algorithm based on modularity optimization, including the local movement, refinement, and aggregation phases.
* **Performance Benchmarking:** Both from-scratch algorithms are benchmarked against their official, optimized library counterparts.
* **Visualization:** Uses `NetworkX` and `Matplotlib` to visualize the detected community clusters and identify high-degree "influencer" nodes.

## Algorithms Implemented

1.  **Infomap (My Contribution):** Implemented in `scratch_infomap.py`. This method views community detection as a data compression problem. It uses a random walker model to find the network partition that provides the most compressed description (shortest codelength) of information flow.
2.  **Leiden (Team Contribution):** Implemented in `scratch_leiden.py`. This algorithm is a refinement of the Louvain method that optimizes a "modularity" score. It guarantees that communities are well-connected and prevents the poorly-connected clusters that Louvain can sometimes produce.

## Dataset

* **Source:** [Stanford SNAP Twitter Dataset](https://snap.stanford.edu/data/ego-Twitter.html)
* **Size:** 81,306 nodes and 1,768,149 edges.
* **Analysis:** Due to the O(n^2) complexity of our from-scratch builds, we ran our final analysis on 200-node subgraphs centered on political news accounts (CNN, The New Yorker, Fox News) to test for echo chambers.

## Key Findings

Our from-scratch `ScratchInfomap` implementation successfully identified 3 distinct communities in the 200-node political subgraph and achieved a **modularity score of 0.5482** (rounded to 0.55).
This is a strong result, as a modularity score greater than 0.3 is generally considered to demonstrate significant community structure. Some really interesting communities can be detected if the graph subset is modified!
