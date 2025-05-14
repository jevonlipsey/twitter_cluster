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

MAX_NODES = 200
SHOW_TOP_USERS = 10
ITERATIONS = 10


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
            module_flow[module] += self.flow[node]

        processed = set()
        for u, v in self.graph.edges():
            if (u, v) in processed or (v, u) in processed:
                continue
            processed.add((u, v))
            mod_u = modules[u]
            mod_v = modules[v]
            if mod_u != mod_v:
                exit_probs[mod_u] += self.flow[u] / self.graph.degree(u)
                exit_probs[mod_v] += self.flow[v] / self.graph.degree(v)

        total_exit = sum(exit_probs.values())

        if total_exit == 0:
            H_P_single = 0.0
            for flow in self.flow.values():
                if flow > 0:
                    H_P_single -= flow * math.log2(flow)
            return H_P_single

        H_Q = -sum((exit / total_exit) * math.log2(exit / total_exit)
                   for exit in exit_probs.values() if exit > 0)

        H_P = 0
        for mod, flow in module_flow.items():
            prob_exit = exit_probs.get(mod, 0)
            prob_stay = flow - prob_exit
            total = flow

            if total > 0:
                terms = []
                if prob_exit > 0:
                    terms.append((prob_exit / total) * math.log2(prob_exit / total))
                if prob_stay > 0:
                    terms.append((prob_stay / total) * math.log2(prob_stay / total))
                H_P_contribution = sum(terms) * flow
                H_P += H_P_contribution

        H_P = -H_P
        L = total_exit * H_Q + H_P
        return L

    def run(self, iterations=10):
        best_modules = self.modules.copy()
        best_score = self._map_equation(best_modules)

        for _ in range(iterations):
            nodes = list(self.graph.nodes())
            random.shuffle(nodes)

            new_modules = self.modules.copy()
            for node in nodes:
                current_module = self.modules[node]
                neighbors = list(self.graph.neighbors(node))
                neighbor_modules = {self.modules[n] for n in neighbors}
                candidate_modules = neighbor_modules.union({current_module, node})

                best_local_score = float('inf')
                best_local_module = current_module

                for candidate in candidate_modules:
                    temp_modules = new_modules.copy()
                    temp_modules[node] = candidate
                    score = self._map_equation(temp_modules)

                    if score < best_local_score:
                        best_local_score = score
                        best_local_module = candidate

                new_modules[node] = best_local_module  # Deferred update

            self.modules = new_modules

            current_score = self._map_equation(self.modules)
            if current_score < best_score:
                best_score = current_score
                best_modules = self.modules.copy()

        self.modules = best_modules
        return self.modules


# helper functions section:
def get_twitter_handle(user_id):
    """converts node id to twitter handle using twitids.com"""
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

    return user_id


def print_summary(G, communities, top_k=10):
    from networkx.algorithms.community.quality import modularity

    # Convert communities to the format expected by networkx (a list of sets)
    module_sets = defaultdict(set)
    for node, community in communities.items():
        module_sets[community].add(node)
    community_list = list(module_sets.values())

    # Print basic stats
    print(f"\n[Summary]")
    print(f"Nodes: {G.number_of_nodes()}, Edges: {G.number_of_edges()}")
    print(f"Communities found: {len(community_list)}")
    print(f"Average community size: {sum(len(c) for c in community_list) / len(community_list):.2f}")

    # Modularity
    mod = modularity(G, community_list)
    print(f"Modularity: {mod:.4f}")

    # Top degree nodes
    print(f"\nTop {top_k} high-degree nodes:")
    top_nodes = sorted(G.degree, key=lambda x: x[1], reverse=True)[:top_k]
    for i, (node, deg) in enumerate(top_nodes, start=1):
        comm = communities.get(node, 'N/A')
        handle = get_twitter_handle(node)
        print(f"{i}. Node {node} (@{handle}) - Degree: {deg}, Community: {comm}")


def dl_graph():
    """pulls graph from google drive and extracts it"""
    output = 'twitter_graph.tar.gz'
    extract_dir = 'twitter_graph'
    GRAPH_DIR = os.path.join(extract_dir, 'twitter')
    url = 'https://drive.google.com/uc?id=172sXL1aeK_ZNXCa3WCjkMqtJsn87SMgx&confirm=t'

    if not os.path.exists(GRAPH_DIR):
        if not os.path.exists(output):
            print("downloading graph data...")
            gdown.download(url, output, quiet=False)
        print("extracting data...")
        with tarfile.open(output, 'r:gz') as tar:
            tar.extractall(extract_dir)
    else:
        print(f"graph directory {GRAPH_DIR} found; using existing data.")

    return GRAPH_DIR


def save_graph(graph, filename):
    """saves a NetworkX graph to an .edges file"""
    with open(filename, 'w') as f:
        for u, v in graph.edges():
            f.write(f"{u} {v}\n")
    print(f"graph edges saved to {filename}")


def build_graph(graph_dir):
    """builds combined graph out of .edges files"""
    edge_files = glob.glob(os.path.join(graph_dir, '*.edges'))
    G = nx.Graph()
    for f in edge_files:
        with open(f, 'r') as file:
            for line in file:
                u, v = line.strip().split()
                G.add_edge(u, v)
    print(f"success: combined graph has {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")
    return G


def build_news_subgraph(graph_dir, ego_ids):
    """builds a filtered subgraph with given ego ids. in our case, used to build subgraphs off news accounts"""
    G_news = nx.Graph()
    for eid in ego_ids:
        ego_file = os.path.join(graph_dir, f"{eid}.edges")
        with open(ego_file, 'r') as f:
            for line in f:
                u, v = line.strip().split()
                G_news.add_edge(u, v)
    print(f"left vs. right news subgraph: {G_news.number_of_nodes()} nodes, {G_news.number_of_edges()} edges")
    return G_news


def build_small_subgraph(G, max_nodes=MAX_NODES):
    """Builds a small ego subgraph with up to `max_nodes` nodes."""
    center_node = random.choice(list(G.nodes()))
    G_small = nx.ego_graph(G, center_node, radius=2)
    print(f"ego subgraph: {G_small.number_of_nodes()} nodes, {G_small.number_of_edges()} edges")

    # If G_small is too big, trim to max_nodes
    if G_small.number_of_nodes() > max_nodes:
        sampled_nodes = random.sample(list(G_small.nodes()), max_nodes)
        G_small = G_small.subgraph(sampled_nodes).copy()
        print(f"trimmed to max_nodes: {G_small.number_of_nodes()} nodes, {G_small.number_of_edges()} edges")
    return G_small


def run_infomap(G_working, iterations=10):
    """runs our ScratchInfomap on the graph and returns communities and timing info."""
    infomap = ScratchInfomap(G_working)
    start_time = time.time()
    communities = infomap.run(iterations)
    end_time = time.time()
    print(f"Infomap run complete: {end_time - start_time:.2f} seconds")
    return communities, start_time, end_time


def plot_communities(G_working, communities):
    unique_communities = set(communities.values())
    community_to_int = {comm: i for i, comm in enumerate(unique_communities)}
    int_communities = {node: community_to_int[comm] for node, comm in communities.items()}
    num_coms = len(unique_communities)
    print(f"Found {num_coms} communities.")

    pos = nx.spring_layout(G_working, seed=42)
    community_map = {node: int_communities[node] for node in G_working.nodes()}
    colors = plt.colormaps.get_cmap('tab20')
    fig, ax = plt.subplots(figsize=(12, 10))
    nx.draw_networkx_edges(G_working, pos, alpha=0.3, ax=ax)

    node_colors = [colors(community_map.get(node, 0) % 20) for node in G_working.nodes()]
    nx.draw_networkx_nodes(G_working, pos, node_color=node_colors, node_size=100, ax=ax)

    top_nodes = sorted(G_working.degree, key=lambda x: x[1], reverse=True)[:SHOW_TOP_USERS]
    indicator_map = {}
    for i, (node, _) in enumerate(top_nodes):
        indicator = str(i + 1)
        indicator_map[node] = indicator
        ax.text(pos[node][0], pos[node][1], indicator, fontsize=9, fontweight='bold',
                ha='center', va='center', color='black')

    legend_data = []
    for i, (node, _) in enumerate(top_nodes):
        indicator = str(i + 1)
        handle = get_twitter_handle(node)
        legend_data.append((indicator, node, handle, community_map.get(node, 'N/A'), G_working.degree[node]))

    ax.set_title("ScratchInfomap Twitter Communities")
    ax.axis('off')
    legend_str = "\n".join([f"{ind}. @{handle}, Com: {com}, Deg: {deg}"
                            for ind, uid, handle, com, deg in legend_data])
    plt.figtext(0.99, 0.5, legend_str, fontsize=9, ha='right', va='center',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.6))
    plt.tight_layout()
    plt.savefig('twitter_communities.png', dpi=300)
    print("Plot saved as 'twitter_communities.png'.")

    community_sizes = Counter(int_communities.values())
    print("\nCommunity sizes:")
    for comm_id, size in sorted(community_sizes.items()):
        print(f"Community {comm_id}: {size} nodes")

    print("\nTop nodes and their communities:")
    for i, (node, _) in enumerate(top_nodes):
        community_id = int_communities.get(node, 'N/A')
        handle = get_twitter_handle(node)
        print(f"Node {node}: Community {community_id}, Handle: @{handle}")


def compare_algs(G_working, scratch_start, scratch_end):
    """compare with official infomap"""
    im = Infomap()
    node_id_map = {node: i for i, node in enumerate(G_working.nodes())}
    for u, v in G_working.edges():
        im.addLink(node_id_map[u], node_id_map[v])
    official_start = time.time()
    im.run()
    official_end = time.time()

    print(f"[[ our implemented Infomap took {scratch_end - scratch_start:.4f} seconds ]]")
    print(f"[[ official Infomap took {official_end - official_start:.4f} seconds ]]")


def main():
    twitter_data = dl_graph()
    ego_ids = ['2097571', '14677919', '1367531']  # CNN, New Yorker, Fox News

    G = build_graph(twitter_data)
    G_news = build_news_subgraph(twitter_data, ego_ids)
    G_small = build_small_subgraph(G_news)

    # save_graph(G_news, 'g_news.edges')
    # use to toggle between graphs
    G_working = G_small

    communities, scratch_start, scratch_end = run_infomap(G_working, ITERATIONS)
    print_summary(G_working, communities)
    plot_communities(G_working, communities)
    compare_algs(G_working, scratch_start, scratch_end)


main()
