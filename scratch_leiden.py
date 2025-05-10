#from infomap import Infomap
import leidenalg
import gdown
import tarfile
import os
import glob
import igraph as ig
import networkx as nx
import matplotlib.pyplot as plt
import random
import copy
from collections import defaultdict

output = 'twitter_graph.tar.gz'
extract_dir = 'twitter_graph'
GRAPH_DIR = os.path.join(extract_dir, 'twitter')
url = 'https://drive.google.com/uc?id=172sXL1aeK_ZNXCa3WCjkMqtJsn87SMgx&confirm=t'

# if path has been downloaded already, use existing data
if not os.path.exists(GRAPH_DIR):
    # check if the tar file exists, download if not
    if not os.path.exists(output):
        print("downloading graph data...")
        gdown.download(url, output, quiet=False)
    # extract data
    print("extracting data...")
    with tarfile.open(output, 'r:gz') as tar:
        tar.extractall(extract_dir)
else:
    print(f"graph directory {GRAPH_DIR} found; using existing data.")

# get all edge files from data
edge_files = glob.glob(os.path.join(GRAPH_DIR, '*.edges'))

# init graph
G = nx.Graph()

# loop through all .edges files and add edges to a combined graph
for f in edge_files:
    with open(f, 'r') as f:
        for line in f:
            u, v = line.strip().split()
            G.add_edge(u, v)

print(f"success: combined graph has {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")



# #convert from nx to igraph
# # take subgraph of 500 nodes from largest connected component
# largest_cc = max(nx.connected_components(G), key=len)
# sample_nodes = random.sample(list(largest_cc), 500)
# G_sub = G.subgraph(sample_nodes)
# g = ig.Graph.from_networkx(G_sub)

# #partition graph with leiden algorithm
# communities = g.community_leiden(objective_function="modularity")

# #coloring nodes according to communities
# num = len(communities)
# pallete = ig.RainbowPalette(n=num)

# for i, community in enumerate(communities):
#   g.vs[community]["color"] = i
#   community_edges = g.es.select(_within=community)
#   community_edges["color"] = i

# #plot graph
# fig, ax = plt.subplots()

# ig.plot(communities,
#     target = ax,
#     mark_groups = True,
#     pallet = pallete,
#     vertex_size = 15,
#     edge_width = 1,)

# print("Leiden Clustering on Twitter Subgraph (500 nodes)")
# fig.set_size_inches(15,15)

# #stats
# print("Number of communities: ", len([community for community in communities if len(community)>1]))
# print("Number of verticies: ", g.vcount())
# print("Modularity: ", communities.modularity)

# # take subgraph of 500 nodes from largest connected component
# largest_cc = max(nx.connected_components(G), key=len)
# sample_nodes = random.sample(list(largest_cc), 500)
# G_sub = G.subgraph(sample_nodes)

# #get nodes from sub graph
# nodes = list(G_sub.nodes())
# # print("List of nodes: ", nodes)

# #communities start as individual nodes
# communities = {node: [node] for node in nodes}
# print("List of orig communities: ", communities)

class Leiden():
    def __init__(self, graph):
        self.graph = graph
        self.communities = {node:node for node in graph.nodes()}


    def run(self):
       self.localMoving()

    def localMoving(self):
        queue = list(self.graph.nodes())
        random.shuffle(queue) # randomize order of nodes
      #for node in queue itterate through neighbors and get neighbor's community
        while queue:
            node = queue.pop(0)
            community = self.communities[node]
            bestCommunity = community
        #get initial modularity 
            modularity = self.modularityGain(self.communities) #########
            betterCommunities = {} # holds neighbors whose communities increased modularity and the quality/modularity score

         #for each neighbor check if modularity is gained when node moved to its neighbor's community
            for neighbor in self.graph.neighbors(node):
                neighborCommunity = self.communities[neighbor]

                #try moving node 
                self.communities[node] = neighborCommunity
                gain = self.modularityGain(self.communities)

            #if modularity is gained then update betterCommunities
                if gain > modularity:
                    betterCommunities[gain] = neighbor

                #revert move 
                self.communities[node] = community
        
        #randomly but weighted by modularity score, select a new community with improved modularity/that increases the quality function
            if betterCommunities:
                coms = [betterCommunities[key] for key in betterCommunities]
                modVals = [key for key in betterCommunities]
            #random choices doesn't evaluate if number is less than or = zero so convert num if less than FIX THIS
                for i, val in enumerate(modVals):
                    if val < 0:
                        modVals[i] = 0.0000000000000000001
                  
                newCommunityNode = random.choices(coms,weights=modVals,k=1)
                newCommunityNode = newCommunityNode[0]

            #since node is moving to new community, add its old neighbors that were not in the new community and not in the queue to the queue
                for neighbor in self.graph.neighbors(node):
                    if self.communities[neighbor] != self.communities[newCommunityNode] and neighbor not in queue:
                        queue.append(neighbor)

            # #move node to new community 
            # communities[newCommunityNode].append(node)
            #set node community to new community
                self.communities[node] = self.communities[newCommunityNode]
            
                  
        self.refinementOfPartition()
      
    def modularityGain(self,communities):
        #dictionary of sets to prevent repeated communities in partition
        partition = defaultdict(set)
        #keys are set to a unique node and values are a set of nodes in their community
        for node, com in communities.items():
            partition[com].add(node)
        #get list of values/communities
        partition = list(partition.values())
        #uses network x for calculate modularity of this partitioning
        return nx.community.modularity(self.graph,partition)
        
    def refinementOfPartition(self):
        #start with all nodes in singleton communitites
        refined = {node: node for node in self.graph.nodes()}
        #dictionary of sets representing comunitiy mapping of nodes from local movement phase
        communities = defaultdict(set)
        for node, com in self.communities.items():
            communities[com].add(node)

    # get all nodes from graph and randomly shuffle their orders
        for node in self.graph.nodes():
            #use dict to get the nodes within the same comm and then filter to only nodes connected to current node
            localNodes = list(communities[node])  
            localNodes = [v for v in localNodes if v in self.graph.neighbors(node)]          
            random.shuffle(localNodes)

            currentCommunity = refined[node] # node's current community
            bestCommunity = currentCommunity #will be set to the communitiy that maximizes modularity
            nodeModularity = self.modularityGain(refined) #original mod

        # iterate through each node to find optimal move
            for localNode in localNodes:  

                localCommunity = refined[localNode]
                # skip if the neighbor is in the same community
                if localCommunity == currentCommunity:
                    continue

                # copy community labels and simulate the move
                refined[node] = localCommunity
                gain = self.modularityGain(refined)

                # if moving the node improves modularity, update bestCommunity
                if gain > nodeModularity:
                    bestCommunity = localCommunity
                #revert move for now
                refined[node] = currentCommunity
            refined[node] = bestCommunity  # move the node to the best community found
        
        #Testing Uncomment to see refined comms
        # r = defaultdict(set)
        # for n, c in self.communities.items():
        #     r[c].add(n)
        # print(r)
            
    def aggregateGraph(self):
        pass

    def addEdges(self):
        pass
        
    def colorCommunities(self):
        def randomRGB(): #helper function to generate random color value for communities
            return(random.random(), random.random(), random.random())
        colorMap = {} # dictionary for communityID colors. keys are communityID, and value is its respective color
        nodeColors = {} # tracks community color for each respective node

        for node in self.graph.nodes():
            communityID = self.communities[node] # finds what community node belongs to
            if communityID not in colorMap:
                colorMap[communityID] = randomRGB() # new key-value pair in colorMap dictionary with communityID and color
            nodeColors[node] = colorMap[communityID] # assigns node color based on community color

        return nodeColors 
   

""""Testing/Running Code"""

## Now repeat the process starting with moveNodes until modularity has been maximized
# take subgraph of 500 nodes from largest connected component
largest_cc = max(nx.connected_components(G), key=len)
sample_nodes = random.sample(list(largest_cc), 500)
G_sub = G.subgraph(sample_nodes)

#get nodes from sub graph
nodes = list(G_sub.nodes())
# print("List of nodes: ", nodes)

#communities start as individual nodes
communities = {node: [node] for node in nodes}
# print("List of orig communities: ", communities)

alg = Leiden(G_sub)
alg.run()
print("Finished!")
