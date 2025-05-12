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
import requests
from collections import defaultdict,Counter
import itertools
from operator import itemgetter

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



#convert from nx to igraph
# take subgraph of 500 nodes from largest connected component
largest_cc = max(nx.connected_components(G), key=len)
sample_nodes = random.sample(list(largest_cc), 1500)
G_sub = G.subgraph(sample_nodes)
g = ig.Graph.from_networkx(G_sub)

#partition graph with leiden algorithm
communities = g.community_leiden(objective_function="modularity")

#coloring nodes according to communities
num = len(communities)
pallete = ig.RainbowPalette(n=num)

for i, community in enumerate(communities):
  g.vs[community]["color"] = i
  community_edges = g.es.select(_within=community)
  community_edges["color"] = i

#plot graph
fig, ax = plt.subplots()

ig.plot(communities,
    target = ax,
    mark_groups = True,
    pallet = pallete,
    vertex_size = 15,
    edge_width = 1,)

print("Leiden Clustering on Twitter Subgraph")
fig.set_size_inches(15,15)

#stats
print("Number of communities: ", len([community for community in communities if len(community)>1]))
print("Number of verticies: ", g.vcount())
print("Modularity: ", communities.modularity)


class Leiden():
    def __init__(self, graph):
        self.structure = None
        self.origG = graph
        self.graph = graph
        self.communities = {node:node for node in graph.nodes()}
        #keep trak of curr modularity
        self.modularity = self.findModularity(self.communities)
        self.P = None #will be set to the original partition created in local moving phase, used in following itterations of aggregrate graph


    def run(self):
       run = True
       trackQuality = [self.modularity]
       print("Original Modularity: ", self.modularity)
       while run:
            self.communities = self.localMoving()
            self.modularity = self.findModularity(self.communities)
            trackQuality.append(self.modularity)
           

            #if modularity score not improving
            if trackQuality[-1] == trackQuality[-2]:
                alg.visualize()
                run = False
                print("Maximum Modularity Reached, Quality Stabalized")
           

    def localMoving(self):
        print("Local moving phase...")
        partition = {node:node for node in self.graph} # reset communities
        queue = list(self.graph.nodes())
        random.shuffle(queue) # randomize order of nodes
      #for node in queue itterate through neighbors and get neighbor's community
        while queue:
            node = queue.pop(0)
            community = partition[node]
            bestCommunity = community
        #get initial modularity 
            modularity = self.findModularity(partition)
            betterCommunities = {} # holds neighbors whose communities increased modularity and the quality/modularity score

         #for each neighbor check if modularity is gained when node moved to its neighbor's community
            for neighbor in self.graph.neighbors(node):
                neighborCommunity = partition[neighbor]

                #try moving node 
                partition[node] = neighborCommunity
                gain = self.findModularity(partition)

            #if modularity is gained then update betterCommunities
                if gain > modularity:
                    betterCommunities[neighbor] = gain

                #revert move 
                partition[node] = community
        
        #randomly but weighted by modularity score, select a new community with improved modularity/that increases the quality function
            if betterCommunities:
                qualityIncrease = [betterCommunities[key] for key in betterCommunities]
                coms = [key for key in betterCommunities]
            # #random choices doesn't evaluate weights if number is less than or = zero so convert all weights to still be representative

                if any(q <= 0.0 for q in qualityIncrease):
                    minWeight = abs(min(qualityIncrease))+1
                    shifted = [q+minWeight for q in qualityIncrease ]
                    qualityIncrease = shifted
                   
                  
                newCommunityNode = random.choices(coms,weights=qualityIncrease,k=1)
                newCommunityNode = newCommunityNode[0]

            #since node is moving to new community, add its old neighbors that were not in the new community and not in the queue to the queue
                for neighbor in self.graph.neighbors(node):
                    if partition[neighbor] != partition[newCommunityNode] and neighbor not in queue:
                        queue.append(neighbor)

            # #move node to new community 
            # communities[newCommunityNode].append(node)
            #set node community to new community
                partition[node] = partition[newCommunityNode]
            
        return self.refinementOfPartition(partition)

    """
    Un-Directed Modularity Formula:
    Def. Modularity is a measure of the structure of networks or graphs which measures the strength of division of a network into modules (also called groups, clusters or communities). Networks with high modularity have dense connections between the nodes within modules but sparse connections between nodes in different modules. (Wiki)

    Q = (1 / 2m) * sum over i,j [ A_ij - (d_i * d_i) / 2m] if c_i == c_j

    Where:

    The sum is over all node pairs (i,j) that are in the same community.
    Q = Modularity
    A_ij = 1 if edge from i to j exists, else 0
    d_i = degree of node i
    d_j = degree of node j
    m = total # of edges in graph
    c_i,c_j = communities of node i and j
    """
    def findModularity(self,partiiton):
        g = self.origG
        m = g.number_of_edges()
        degree = dict(g.degree())

        #community to nodes mapping
        comToNodes = defaultdict(set)
        for node,com in partiiton.items():
            comToNodes[com].add(node)

        Q = 0.0
        for nodes in comToNodes.values():
            for i in nodes:
                for j in nodes:
                    A_ij = 1 if g.has_edge(i,j) else 0
                    expected = (degree[i]*degree[j])/ (2*m)
                    Q += A_ij - expected
    
        return Q/(2*m)

        
    def refinementOfPartition(self,partition):
        print("Refinement phase...")
        #start with all nodes in singleton communitites
        refined = {node: node for node in self.graph.nodes()}
        #dictionary of sets representing comunitiy mapping of nodes from local movement phase
        communities = defaultdict(set)
        for node, com in partition.items():
            communities[com].add(node)


    # get all nodes from graph and randomly shuffle their orders
        for node in self.graph.nodes():
            #use dict to get the nodes within the same comm and then filter to only nodes connected to current node

            localNodes = list(itertools.chain.from_iterable([communities[v] for v in communities if node in communities[v]]) )    
            localNodes = [localNodes[i] for i in range(len(localNodes)) if localNodes[i] in self.graph.neighbors(node)]
            random.shuffle(localNodes)

            currentCommunity = partition[node] # node's current community
            bestCommunity = currentCommunity #will be set to the communitiy that maximizes modularity
            nodeModularity = self.findModularity(partition) #original mod

        # iterate through each node to find optimal move
            for localNode in localNodes:  

                localCommunity = refined[localNode]
                # skip if the neighbor is in the same community
                if localCommunity == currentCommunity:
                    continue

                # copy community labels and simulate the move
                refined[node] = localCommunity
                gain = self.findModularity(refined)

                # if moving the node improves modularity, update bestCommunity
                if gain > nodeModularity:
                    bestCommunity = localCommunity
                #revert move for now
                refined[node] = currentCommunity
            refined[node] = bestCommunity  # move the node to the best community found
        
        return self.aggregateGraph(refined)
            
    def aggregateGraph(self,refined):
        print("Aggregrating graph...")
        aggregatedGraph = nx.Graph()
        communityNodes = defaultdict(set)
        edgeWeights = {}
        
        for node, community in refined.items(): # make key-pair values of communities and nodes to communityNodes
            communityNodes[community].add(node) 
        
        communityStructure = {}

        for community in communityNodes: # add nodes to aggregatedGraph based on total communities
            aggregatedGraph.add_node(community,)
            communityStructure[community] = communityNodes[community]
          
        self.structure = communityStructure
            

        for rootNode, targetNode in self.graph.edges(): # count edges in communities
            rootNodeComm = refined[rootNode]
            targetNodeComm = refined[targetNode]

            if rootNodeComm != targetNodeComm: # skip iteration if part of the same community
                edgeKey = tuple(sorted((rootNodeComm, targetNodeComm))) # creates key of two communities that share edge
                if edgeKey not in edgeWeights:
                    edgeWeights[edgeKey] = 0
                edgeWeights[edgeKey] += 1 

        for (rootNodeComm, targetNodeComm), count in edgeWeights.items(): # add edges between communities
            aggregatedGraph.add_edge(rootNodeComm, targetNodeComm, weight = count) # weight is how often the community appeared
        
        self.graph = aggregatedGraph # make original graph the new aggregatedGraph
        return refined
         

    def addEdges(self, edgeList): # input list of node pairs using ".edges()"
        for sourceNode, targetNode in edgeList: 
            self.graph.add_edge(sourceNode, targetNode) # adds edge for each respective node pair
        
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

        return list(nodeColors.values())
    
    def visualize(self):
        
        print("Num of Communities/Nodes: ", self.graph.number_of_nodes())
        print("Num of Edges: ", self.graph.number_of_edges())
        # print("Neighbors: ", [(node, list(self.graph.neighbors(node)))for node in self.graph.nodes() if len(list(self.graph.neighbors(node))) > 0 ])
        # print("Edges: ", list(self.graph.edges()))
        print("Modularity: ", self.findModularity(self.communities))
        # print("Structure of Aggregated Communities: ",[(key,self.structure[key]) for key in self.structure if len(self.structure[key])>1 ])

        nodeColors = self.colorCommunities()

        plt.figure(figsize=(10, 10))
        plt.tight_layout()
        pos = nx.spring_layout(self.graph, seed=42)
        nx.draw(
            self.graph, pos, node_color=nodeColors,
            node_size=50, font_size=10, font_color='black', edge_color='gray'
            )

        plt.title("Aggregated Graph by Community", fontsize=14)
        plt.show()
        

def get_twitter_handle(user_id): # userID retrieval from twitter (same code as InfoMap)
    url = 'https://twitids.com/'  
    session = requests.Session()
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) '
                'Chrome/91.0.4472.124 Safari/537.36',
        'Accept': 'application/json',
    }
    form_data = {'user_input': user_id}

    try:
        response = session.post(url, data=form_data, headers=headers, timeout=10)
        if response.status_code == 200:
            data = response.json()
            if 'screen_name' in data:
                return data['screen_name']
            else:
                print(f"!! 'screen_name' not found in JSON for {user_id}")
        else:
            print(f" !! HTTP error {response.status_code} for {user_id}")
    except Exception as e:
        print(f" !! Exception occurred for {user_id}: {e}")
        
    return user_id  # returns id as a fallback, could be an old/changed account

   

""""Testing/Running Code"""

# ## Now repeat the process starting with moveNodes until modularity has been maximized
# # take subgraph of 500 nodes from largest connected component
# largest_cc = max(nx.connected_components(G), key=len)
# sample_nodes = random.sample(list(largest_cc), 1000)
# G_sub = G.subgraph(sample_nodes)

# #get nodes from sub graph
# nodes = list(G_sub.nodes())
# # print("List of nodes: ", nodes)

# #communities start as individual nodes
# communities = {node: [node] for node in nodes}
# # print("List of orig communities: ", communities)

# """TESTING ONLY delete later"""
# # # Create a simple graph with 15 nodes and some edges connecting them
# edges = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 0),
#     (6, 7), (7, 8), (8, 9), (9, 6), (3, 9), (5, 6),
#     (10, 11), (11, 12), (12, 13), (13, 14), (14, 10), (1, 11)]

# # Create the graph and add the edges
# graph = nx.Graph()
# graph.add_edges_from(edges)

# # # Improve the visualization with a layout and better styling
plt.figure(figsize=(8, 8))
pos = nx.spring_layout(G_sub, seed=42)  # Use a seed for consistent layout

# Draw the graph with additional details
nx.draw(
    G_sub, pos, node_color='lightblue',
    node_size=50, font_size=10, font_color='black', edge_color='gray'
)

# Show the graph
plt.title("Graph Visualization", fontsize=14)
plt.show()

alg = Leiden(G_sub)
alg.run()

