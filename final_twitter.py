import glob
import math
import os
import random
import tarfile
from collections import defaultdict, Counter
import gdown
import matplotlib.pyplot as plt
import networkx as nx
import requests
import time
from infomap import Infomap

MAX_NODES = 1000
SHOW_TOP_USERS = 5


class ScratchInfomap:
    def __init__(self, graph):
        self.graph = graph  # networkx.Graph
        self.modules = {node: node for node in graph.nodes()}
        self.flow = self._compute_flow()

    def _compute_flow(self):
        total_degree = sum(dict(self.graph.degree()).values())
        return {node: self.graph.degree(node) / total_degree for node in self.graph.nodes()}

    def _map_equation(self, modules):
        exit_probs = defaultdict(float)
        module_flow = defaultdict(float)

        for node, module in modules.items():
            node_flow = self.flow[node]
            module_flow[module] += node_flow
            for neighbor in self.graph.neighbors(node):
                if modules[neighbor] != module:
                    exit_probs[module] += node_flow / self.graph.degree(node)

        total_exit = sum(exit_probs.values())
        H_Q = -sum((exit / total_exit) * math.log2(exit / total_exit)
                   for exit in exit_probs.values() if exit > 0)

        H_P = 0
        for mod, flow in module_flow.items():
            prob_exit = exit_probs[mod]
            prob_stay = flow - prob_exit
            total = prob_exit + prob_stay
            if total > 0:
                H_P += total * -(
                    (prob_exit / total) * math.log2(prob_exit / total) if prob_exit > 0 else 0 +
                                                                                             (
                                                                                                     prob_stay / total) * math.log2(
                        prob_stay / total) if prob_stay > 0 else 0
                )

        return total_exit * H_Q + H_P

    def run(self, iterations=10):
        for _ in range(iterations):
            nodes = list(self.graph.nodes())
            random.shuffle(nodes)
            for node in nodes:
                best_module = self.modules[node]
                best_score = self._map_equation(self.modules)

                neighbor_modules = {self.modules[n] for n in self.graph.neighbors(node)}
                for mod in neighbor_modules:
                    old_mod = self.modules[node]
                    self.modules[node] = mod
                    score = self._map_equation(self.modules)
                    if score < best_score:
                        best_score = score
                        best_module = mod
                    self.modules[node] = old_mod

                self.modules[node] = best_module

        return self.modules


# helper functions section:
# convert node id's to handles
def get_twitter_handle(user_id):
    url = 'https://twitids.com/'  #
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
##


# init graph section:
# download graph data and init as nx
output = 'twitter_graph.tar.gz'
extract_dir = 'twitter_graph'
GRAPH_DIR = os.path.join(extract_dir, 'twitter')
url = 'https://drive.google.com/uc?id=172sXL1aeK_ZNXCa3WCjkMqtJsn87SMgx&confirm=t'
edges_list = []

if not os.path.exists(GRAPH_DIR):
    if not os.path.exists(output):
        print("downloading graph data...")
        gdown.download(url, output, quiet=False)
    print("extracting data...")
    with tarfile.open(output, 'r:gz') as tar:
        tar.extractall(extract_dir)
else:
    print(f"graph directory {GRAPH_DIR} found; using existing data.")

edge_files = glob.glob(os.path.join(GRAPH_DIR, '*.edges'))

G = nx.Graph()
for f in edge_files:
    with open(f, 'r') as f:
        for line in f:
            u, v = line.strip().split()
            G.add_edge(u, v)
            edges_list.append((u, v))

print(f"success: combined graph has {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")
##

# init subgraphs section: test with subgraphs until ready to build full graph (if possible)
# build random ego as a g_sub
# use Fox News ego graph specifically
# load ego graphs for CNN, New Yorker, and Fox News
ego_ids = ['2097571', '14677919', '1367531']  # CNN, New Yorker, Fox News
G_sub = nx.Graph()

for eid in ego_ids:
    ego_file = os.path.join(GRAPH_DIR, f"{eid}.edges")
    with open(ego_file, 'r') as f:
        for line in f:
            u, v = line.strip().split()
            G_sub.add_edge(u, v)

print(f"Combined subgraph: {G_sub.number_of_nodes()} nodes, {G_sub.number_of_edges()} edges")

# run Infomap section:
# set g_working to G for main graph or G_sub2
G_working = G_sub
#nG_working = G
infomap = ScratchInfomap(G_working)
scratch_start = time.time()
communities = infomap.run()
scratch_end = time.time()
print("infomap successfully detected communities")

# map communities to ints
unique_communities = set(communities.values())
community_to_int = {comm: i for i, comm in enumerate(unique_communities)}
int_communities = {node: community_to_int[comm] for node, comm in communities.items()}
num_coms = len(unique_communities)
print(f"Found {num_coms} communities")
##

# visualization section:
pos = nx.spring_layout(G_working, seed=42)
community_map = {}
for node, com_id in int_communities.items():
    community_map[node] = com_id

colors = plt.colormaps.get_cmap('tab20')
fig, ax = plt.subplots(figsize=(12, 10))

# draw edges
nx.draw_networkx_edges(G_working, pos, alpha=0.3, ax=ax)
# draw nodes
node_colors = [colors(community_map.get(node, 0) % 20) for node in G_working.nodes()]
nx.draw_networkx_nodes(G_working, pos, node_color=node_colors, node_size=100, ax=ax)

# draw top user node indicators
top_nodes = sorted(G_working.degree, key=lambda x: x[1], reverse=True)[:SHOW_TOP_USERS]
indicator_map = {}
for i, (node, _) in enumerate(top_nodes):
    indicator = str(i + 1)
    indicator_map[node] = indicator
    ax.text(pos[node][0], pos[node][1], indicator, fontsize=9, fontweight='bold', ha='center', va='center',
            color='black')

# create legend data with twitter handles
legend_data = []
for i, (node, _) in enumerate(top_nodes):
    indicator = str(i + 1)
    handle = get_twitter_handle(node)
    legend_data.append((indicator, node, handle, community_map.get(node, 'N/A'), G_working.degree[node]))
# draw text legend at right of plot
ax.set_title("Twitter Network Communities")
ax.axis('off')

legend_str = "\n".join([f"{ind}. @{handle}, Com: {com}, Deg: {deg}"
                        for ind, uid, handle, com, deg in legend_data])


plt.figtext(0.99, 0.5, legend_str, fontsize=9, ha='right', va='center',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.6))

plt.tight_layout()
#plt.show() # not working and not sure why... savefig works though
plt.savefig('twitter_communities.png', dpi=300)
print("Plot saved as 'twitter_communities.png'.")
##

# community stats
community_sizes = Counter(int_communities.values())
print("\nCommunity sizes:")
for comm_id, size in sorted(community_sizes.items()):
    print(f"Community {comm_id}: {size} nodes")

print("\ntop nodes and their communities:")
for i, (node, _) in enumerate(top_nodes):
    community_id = int_communities.get(node, 'N/A')
    handle = get_twitter_handle(node)
    print(f"Node {node}: Community {community_id}, Handle: @{handle}")

# time stats
print(f"[[ our implemented Infomap took {scratch_end - scratch_start:.4f} seconds ]]")
print("running official Infomap...")
im = Infomap()
node_id_map = {node: i for i, node in enumerate(G_working.nodes())}
for u, v in G_working.edges():
    im.addLink(node_id_map[u], node_id_map[v])
start = time.time()
im.run()
end = time.time()
print(f"[[ official Infomap took {end - start:.4f} seconds ]]")
