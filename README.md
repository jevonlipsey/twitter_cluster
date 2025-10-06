# Twitter Community Detection
A comparative study of community detection algorithms on Twitter social networks using Leiden and Infomap algorithms.

## Overview
This project implements and compares two community detection algorithms to identify social communities within Twitter networks:

Leiden Algorithm: Optimizes modularity through local movement, refinement, and aggregation phases
Infomap Algorithm: Uses information compression and random walk flow to detect communities

## Key Features
Custom implementations of both Leiden and Infomap algorithms from scratch
Analysis of Twitter subgraphs centered around political news accounts (CNN, Fox News, The New Yorker)
Community structure visualization and echo chamber detection
Performance comparison between custom implementations and established libraries

## Results
Successfully identified meaningful political communities in Twitter networks
Demonstrated echo chamber detection capabilities
Achieved modularity scores up to 0.55 with Infomap on smaller subgraphs
Found evidence of cross-community interactions, suggesting less strict echo chambers than expected

## Dataset
Twitter follower network data from Stanford SNAP dataset (80K nodes, 1.2M edges)

## Credits

Built by Jevon Lipsey, Zaharalita Love, and Marcos Arnold on the [SNAP dataset](https://snap.stanford.edu/data/ego-Twitter.html).

