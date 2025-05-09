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

"""
Calculates modularity gain, Check if modularity is increased or decrease when
node is moved to new cluster(newC). Used to check if node should be moved to new cluster"""
def modularityGain(graph, communities, node, newCommunity, initial = True):
  
  partition = [node]
  for neighbor in graph.neighbors(node):
     if communities[neighbor] == newCommunity:
        partition.append(neighbor)
  p = []
  for key in communities:
     if communities[key] not in p:
        p.append(communities[key])

  if not initial:
    for ls in p:
        for n in partition:
            if n in ls:
                 print("Removed: ",n)
                 ls.remove(n)   
    p.append([n for n in partition])
    p = [sublist for sublist in p if sublist]
    print("Pn len:",len(p))
    print("Node: ",node)
    print("Node Community: ",communities[node])
  print()      
  return nx.community.modularity(graph,p)


"""
Iterates through nodes moving them to new clusters to maximize modularity.
Performs this step some arbitrary number of times n.
"""
def moveNodes(graph,communities,queue,n):
    #for range n
    for _ in range(n):

      random.shuffle(queue)
      #for node in nodes itterate through neighbors and set neighbors community
      while queue:
         node = queue.pop(0)
         community = communities[node]
         #get initial modularity 
         temp = copy.deepcopy(communities)
         modularity = modularityGain(graph,temp,node,community)
         betterCommunities = {} #communities that increased modularity and the gain

         #check if modularity is gained if node moved to neighbor community(are there more internal edges
         for neighbor in graph.neighbors(node):
            neighborCommunity = communities[neighbor]
            temp = copy.deepcopy(communities)
            gain = modularityGain(graph,temp,node,neighborCommunity,False)
            #if modularity if gained then move node to target community and track current modularity
            if gain > modularity:
               betterCommunities[gain] = neighbor
        
        #randomly but weighted by modularity score, select a new community with improved modularity/that increases the quality function
         if betterCommunities:
            coms = [betterCommunities[key] for key in betterCommunities]
            modVals = [key for key in betterCommunities]
            for i, val in enumerate(modVals):
               if val < 0:
                  modVals[i] = val*-1
                  
             
            print("Moving to new com: ", coms,modVals)
            newCommunityNode = random.choices(coms,weights=modVals,k=1)
            newCommunityNode = newCommunityNode[0]

            #since node is moving to new community, add its old neighbors that were not in the new community and not in the queue to the queue
            for neighbor in graph.neighbors(node):
                if communities[neighbor] != communities[newCommunityNode] and neighbor not in queue:
                   queue.append(neighbor)
            #move node to new community 
            communities[newCommunityNode].append(node)
            #set node community to new community
            communities[node] = communities[newCommunityNode]
            #remove node from old community, if community only consisted of node then delete it 
            # if len(communities[node]) > 1:
            #      communities[node].remove(node)
            # else:
            #    del communities[node]
                  
    
    print("List of communities after local move: ",communities)
    return communities
         

def refinementOfPartition(graph,communities):
   refined = {}
   for node in communities:
        p_refined = {node: node for node in communities.keys()}
        queue = random.shuffle(communities[node])
      #for node in nodes itterate through neighbors and set neighbors community
        while queue:
         node = queue.pop(0)
         community = communities[node]
         #get initial modularity 
         modularity = modularityGain(graph,communities,node,community)
         betterCommunities = {} #communities that increased modularity and the gain

         #check if modularity is gained if node moved to neighbor community(are there more internal edges
         for n in p_refined:
            neighborCommunity = p_refined[n]
            gain = modularityGain(graph,communities,node,neighborCommunity)
            #if modularity is gained then move node to target community and track current modularity
            if gain > modularity:
               betterCommunities[neighborCommunity] = gain
        
        #randomly but weighted by modularity score, select a new community with improved modularity/that increases the quality function
         if betterCommunities:
            newCommunity = random.choices(betterCommunities.keys(),weights=betterCommunities.values(),k=1)
        
            p_refined[node] =  p_refined[node] + newCommunity

         else:
           p_refined[node] = communities[node]
        
        refined.append(p_refined.values())

   print("List of communities after refinement:",refined)

"""
 Colors the nodes according to the cluster they belong to. Returns list of node
 colors for visual
"""
def colorCommunities(communities,nodes):
  pass

"""
Returns an aggregated graph with communities represented as nodes.
Returns dict of communities to unique nodes in graph
"""
def makeAggregatedGraph(communities):
  pass

"""
Helper func for making aggregated graph, adds edges between nodes if their
 communities were connected but not the same in orig graph.

Returns graph with edges
"""
def add_edges(aggregatedGraph):
  pass


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
print("List of orig communities: ", communities)

moveNodes(G_sub,communities,nodes,5)
