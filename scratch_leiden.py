# === Library Imports ===
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
import time
import requests
from collections import defaultdict,Counter
import itertools
from operator import itemgetter
from collections import defaultdict


# === Accessing SNAP (Stanford Network Analysis Project) Twitter Dataset === 
output = 'twitter_graph.tar.gz'
extract_directory = 'twitter_graph'
graph_directory = os.path.join(extract_directory, 'twitter')
url = 'https://drive.google.com/uc?id=172sXL1aeK_ZNXCa3WCjkMqtJsn87SMgx&confirm=t'

# if path has been downloaded already, use existing data
if not os.path.exists(graph_directory):
    # check if the tar file exists, download if not
    if not os.path.exists(output):
        print("downloading graph data...")
        gdown.download(url, output, quiet =False)
    # extract data
    print("extracting data...")
    with tarfile.open(output, 'r:gz') as tar:
        tar.extractall(extract_directory)
else:
    print(f"graph directory {graph_directory} found; using existing data.")

# get all edge files from data
edge_files = glob.glob(os.path.join(graph_directory, '*.edges'))

# initiate graph
G = nx.Graph()

# loop through all .edges files and add edges to a combined graph
for f in edge_files:
    with open(f, 'r') as f:
        for line in f:
            u, v = line.strip().split()
            G.add_edge(u, v)

print(f"success: combined graph has {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")


# === Scratch Leiden Algorithm Implementation ===
class Leiden():
    def __init__(self, graph):
        
        self.origG = graph
        self.graph = graph
        self.communities = {node:node for node in graph.nodes()}
        #keep track of curr modularity
        self.modularity = self.findModularity(self.communities)
       

    def run(self, resolutionParameter = 1):
       startTime = time.time()
      
    #    trackQuality = [self.modularity]
       print("Original Modularity: ", self.modularity)
       for _ in range(resolutionParameter):
            self.communities = self.localMoving()
            self.modularity = self.findModularity(self.communities)
            # trackQuality.append(self.modularity)

            # #if modularity score not improving
            # if trackQuality[-1] == trackQuality[-2]:
       
       endTime = time.time()
                # # alg.visualize()
                # run = False
                # print("Maximum Modularity Reached, Quality Stabalized")
       print("Runtime: ", endTime - startTime)
       print("Final modularity: ", self.modularity)
       return self.modularity
           

    def localMoving(self, partition=None, agg=False):
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

        if agg:
            #self.visualize(partition,"Local Movement of Aggragated Graph")
            return partition
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
        g = self.graph
        m = g.size(weight="weight")
        if m == 0:
            return 0
        degree = dict(g.degree(weight="weight"))

        #community to nodes mapping
        comToNodes = defaultdict(set)
        for node,com in partiiton.items():
            comToNodes[com].add(node)

        Q = 0.0
        for nodes in comToNodes.values():
           for i,j in itertools.combinations(nodes,2):
                A_ij = g[i][j].get("weight",1) if g.has_edge(i,j) else 0
                d_i = degree.get(i,1)
                d_j = degree.get(j,1)
                if d_i == 0 or d_j == 0:
                    expected = 0
                else:
                    expected = (d_i*d_j/ (2*m))
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
        self.visualize(refined, "Scratch Leiden After Local Movement and Refinement", show_legend=True, get_handle_func=get_twitter_handle)
        return self.aggregateGraph(refined)
            
    def aggregateGraph(self,refined):
        print("Aggregrating graph...")
        aggregatedGraph = nx.Graph()
        communityNodes = defaultdict(set)
        edgeWeights = {}
        partition = {}
        for node, community in refined.items(): # make key-pair values of communities and nodes to communityNodes
            communityNodes[community].add(node) 
        
        for community in communityNodes: # add nodes to aggregatedGraph based on total communities
            aggregatedGraph.add_node(community,)

        for rootNode, targetNode in self.graph.edges(): # count edges in communities
            rootNodeComm = refined[rootNode]
            targetNodeComm = refined[targetNode]

            if rootNodeComm != targetNodeComm: # skip iteration if part of the same community
                edgeKey = tuple(sorted((rootNodeComm, targetNodeComm))) # creates key of two communities that share edge
                if edgeKey not in edgeWeights:
                    edgeWeights[edgeKey] = 1
                edgeWeights[edgeKey] += 1 


        for (rootNodeComm, targetNodeComm), count in edgeWeights.items(): # add edges between communities
            aggregatedGraph.add_edge(rootNodeComm, targetNodeComm, weight=count) # weight is how often the community appeared
        
        for edge in edgeWeights:
            if edge[0] not in partition.keys():
                partition[edge[0]] = edge[1]
                if edge[1] not in partition.keys():
                    partition[edge[1]] = edge[0]
            else:
                partition[edge[1]] = edge[0]
        
        for node in communityNodes.keys():
            if node not in partition.keys():
                partition[node]=node
        
            
        self.graph = aggregatedGraph # make original graph the new aggregatedGraph
        #self.visualize(partition,"Scratch Leiden After Aggregation")

        return self.localMoving(partition,True)
    
     
    def colorCommunities(self,partiton):
        if len(partiton) == 0:
            return ['blue']
        def randomRGB(): #helper function to generate random color value for communities
            return(random.random(), random.random(), random.random())
        colorMap = {} # dictionary for communityID colors. keys are communityID, and value is its respective color
        nodeColors = {} # tracks community color for each respective node

        for node in self.graph.nodes():
            communityID = partiton[node] # finds what community node belongs to
            if communityID not in colorMap:
                colorMap[communityID] = randomRGB() # new key-value pair in colorMap dictionary with communityID and color
            nodeColors[node] = colorMap[communityID] # assigns node color based on community color

        return list(nodeColors.values())
    
    def visualize(self, partition, title, show_legend=False, get_handle_func=None, leaderboard_size=10):
        print("Num of Communities/Nodes: ", self.graph.number_of_nodes())
        print("Num of Edges: ", self.graph.number_of_edges())
        print("Modularity: ", self.findModularity(partition))
        
        nodeColors = self.colorCommunities(partition)
        plt.figure(figsize=(15, 15))
        plt.tight_layout()
        pos = nx.spring_layout(self.graph, seed=42)

        nx.draw(
            self.graph, pos, node_color=nodeColors,
            node_size=50, font_size=10, font_color='black', edge_color='gray'
        )

        plt.title(title, fontsize=16)

        if show_legend and get_handle_func:
            degrees = dict(self.graph.degree())
            top_nodes = sorted(degrees.items(), key=lambda x: x[1], reverse=True)[:leaderboard_size]
            top_nodes = [(i + 1, node, deg, partition.get(node, -1)) for i, (node, deg) in enumerate(top_nodes)]

            legend_lines = ["Top Twitter Users (Scratch Leiden):",
                            "Rank | Node ID      | Handle         | Degree | Comm",
                            "-" * 50]

            for rank, node, deg, comm_id in top_nodes:
                handle = get_handle_func(node)
                handle = '@' + handle if not handle.startswith('@') else handle
                legend_lines.append(f"{rank:<4} | {node:<12} | {handle:<14} | {deg:<6} | {comm_id}")

            full_text = "\n".join(legend_lines)
            plt.gcf().subplots_adjust(left=0.35)  # Shrink plot from the left
            plt.gcf().text(0.02, 0.95, full_text,
                fontsize=9, family='monospace',
                verticalalignment='top', horizontalalignment='left',
                bbox=dict(facecolor='white', edgecolor='black', alpha=0.9))


            # Add numbered labels to top nodes
            for rank, node, deg, comm_id in top_nodes:
                if node in pos:
                    x, y = pos[node]
                    plt.text(
                        x, y, str(rank),
                        fontsize=10, fontweight='bold', color='black',
                        horizontalalignment='center',
                        verticalalignment='center',
                        bbox=dict(facecolor='white', edgecolor='black', boxstyle='circle', alpha=0.7)
                    )

        plt.show()

        
# === Testing + Graphing Section ===

# Function to retrieve Twitter handles from user ID 
def get_twitter_handle(user_id): 
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
        
    return user_id  # Returns user ID as a fallback (could be an old/changed account)

# Function to display top "N" users by degree in the legend

def topGlobalUsersLegend(g, communities, getTwitterHandle, layout, ax=None, leaderboardPositions=5, pallete=None):
    degrees = g.degree()
    node_ids = g.vs["_nx_name"]
    nodeToCommunity = {}

    # Build node-to-community map
    for i, community in enumerate(communities):
        for idx in community:
            node_name = g.vs[idx]["_nx_name"]
            nodeToCommunity[node_name] = i

    community_ids = [nodeToCommunity.get(node_name, -1) for node_name in node_ids]
    full_data = list(zip(node_ids, degrees, community_ids))

    # Get top N nodes by degree and assign rank
    top_nodes_raw = sorted(full_data, key=lambda x: x[1], reverse=True)[:leaderboardPositions]
    top_nodes = [(i + 1, node_name, deg, comm_id) for i, (node_name, deg, comm_id) in enumerate(top_nodes_raw)]

    # Build text-based legend
    legend_lines = ["Top Twitter Users (Entire Graph):", 
                    "Rank | Node ID      | Handle         | Degree | Comm", 
                    "-" * 50]

    node_handle_map = {}
    for rank, node_name, deg, comm_id in top_nodes:
        handle = getTwitterHandle(node_name)
        handle = '@' + handle if not handle.startswith('@') else handle
        node_handle_map[node_name] = handle
        line = f"{rank:<3} | {node_name:<12} | {handle:<14} | {deg:<6} | {comm_id}"
        legend_lines.append(line)

    full_text = "\n".join(legend_lines)

    # Adjust figure to make room for legend on the left
    plt.gcf().subplots_adjust(left=0.3)  
    plt.gcf().text(0.05, 0.95, full_text,
        fontsize=9, family='monospace',
        verticalalignment='top', horizontalalignment='left',
        bbox=dict(facecolor='white', edgecolor='black', alpha=0.9))

    # Add numbered labels to the top nodes
    used_labels = set()
    name_to_idx = {g.vs[i]["_nx_name"]: i for i in range(g.vcount())}

    for rank, node_name, deg, comm_id in top_nodes:
        if node_name in used_labels or node_name not in name_to_idx:
            continue
        used_labels.add(node_name)

        idx = name_to_idx[node_name]
        x, y = layout[idx]

        plt.text(
            x, y, str(rank),
            fontsize=10, fontweight='bold', color='black',
            horizontalalignment='center',
            verticalalignment='center',
            bbox=dict(facecolor='white', edgecolor='black', boxstyle='circle', alpha=0.7)
        )

    return top_nodes




# === Ego Graph from CNN, New Yorker, and Fox News ===
ego_ids = ['2097571', '14677919', '1367531']  # CNN, New Yorker, Fox News- for each index respectfully
G_sub = nx.Graph()

for eid in ego_ids:
    ego_file = os.path.join(graph_directory, f"{eid}.edges")
    with open(ego_file, 'r') as f:
        for line in f:
            u, v = line.strip().split()
            G_sub.add_edge(u, v)

print(f"Combined ego subgraph: {G_sub.number_of_nodes()} nodes, {G_sub.number_of_edges()} edges")

MAX_NODES = 200

if G_sub.number_of_nodes() > MAX_NODES:
    sampled_nodes = random.sample(list(G_sub.nodes()), MAX_NODES)
    G_sub2 = G_sub.subgraph(sampled_nodes).copy()
else:
    G_sub2 = G_sub

print(f"Trimmed ego subgraph (G_sub2): {G_sub2.number_of_nodes()} nodes, {G_sub2.number_of_edges()} edges")

# Function for Leiden Algorithm x Ego Graph Implementation
def leidenEgoGraph():
    g = ig.Graph.from_networkx(G_sub)
    communities = g.community_leiden(objective_function="modularity")
    idx_to_node = g.vs["_nx_name"]

    # Map community membership back to networkx nodes
    membership = {}
    for i, community in enumerate(communities):
        for idx in community:
            node_name = idx_to_node[idx]
            membership[node_name] = i
    nx.set_node_attributes(G_sub, membership, "community")

    # Generate matplotlib layout and colors
    pos = nx.spring_layout(G_sub, seed=42)
    node_colors = [membership[node] for node in G_sub.nodes()]

    fig, ax = plt.subplots(figsize=(15, 15))

    nx.draw(
        G_sub, pos,
        node_color=node_colors,
        cmap=plt.cm.rainbow,
        node_size=50,
        edge_color='gray',
        with_labels=False,
        ax=ax
    )
    plt.title("Leiden Communities on Twitter Ego Subgraph", fontsize=16)

    # Add text-based legend showing top users
    layout = g.layout("fr")
    top_nodes = topGlobalUsersLegend(g,communities,get_twitter_handle,layout=layout,ax=ax, leaderboardPositions=5)
    for rank, node, deg, comm_id in top_nodes:
        x, y = pos[node]
        plt.text(
            x, y, str(rank),
            fontsize=10, fontweight='bold', color='black',
            horizontalalignment='center',
            verticalalignment='center',
            bbox=dict(facecolor='white', edgecolor='black', boxstyle='circle', alpha=0.7)
        )

    plt.show()

    # Statistics
    print("Leiden Clustering on Twitter Subgraph")
    print("Number of communities:", len([c for c in communities if len(c) > 1]))
    print("Number of vertices:", g.vcount())
    print("Modularity:", communities.modularity)

# Function for SCRATCH Leiden Algorithm x Ego Graph Implementation
def scratchEgoGraph():
    alg = Leiden(G_sub2)
    alg.run(resolutionParameter=1)
    alg.visualize(alg.communities, "Scratch Leiden Communities on Twitter Ego Subgraph", show_legend=True, leaderboard_size=10)

    # Statistics
    print("Scratch Leiden Clustering on Ego Graph")
    print("Number of communities:", len(set(alg.communities.values())))
    print("Number of vertices:", len(alg.graph.nodes()))
    print("Final modularity:", alg.modularity)

""""Testing/Running Scratch Leiden"""
mScores = {}
for i in range(1,2):
     alg = Leiden(G_sub2)
     modularity = alg.run(i)
     mScores[i] = modularity
bestMScore = max(mScores.values())
resolutionP = [key for key in mScores if mScores[key] == bestMScore]
resolutionP = resolutionP[0]

print()
alg = Leiden(G_sub2)
print("Number of Iterations: ", resolutionP)
alg.run(resolutionP)
print()

# === Visualize Graph ===
leidenEgoGraph()
scratchEgoGraph()
