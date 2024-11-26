import numpy as np
import networkx as nx
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Set
from dataclasses import dataclass
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
import community  # python-louvain package for community detection
import warnings
from collections import defaultdict

warnings.filterwarnings('ignore')


@dataclass
class ClusteredModelParameters:
    """Parameters for the clustered epidemic model"""
    infection_prob: float
    num_simulations: int
    cluster_simulations: int  # Number of simulations to run per cluster
    seed: Optional[int] = None
    cache_threshold: float = 0.8  # Threshold for using cached results


class ClusteredEpidemicNetwork:
    """Enhanced implementation with clustering and result caching"""

    def __init__(self, G: nx.Graph, params: ClusteredModelParameters):
        """
        Initialize the clustered epidemic model

        Args:
            G: NetworkX graph representing the contact network
            params: ClusteredModelParameters instance with simulation parameters
        """
        # Relabel nodes to integers for consistent processing
        self.node_mapping = {node: i for i, node in enumerate(G.nodes())}
        self.reverse_mapping = {i: node for node, i in self.node_mapping.items()}
        self.G = nx.relabel_nodes(G, self.node_mapping)

        self.params = params
        self.n_nodes = G.number_of_nodes()
        self.infection_times = None
        self.covering_times = None

        # New attributes for clustering
        self.clusters = None
        self.cluster_cache = {}
        self.bridge_edges = set()
        self.internal_edges = defaultdict(set)

        if params.seed is not None:
            np.random.seed(params.seed)

        # Initialize clusters and caches
        self._initialize_clusters()

    def _initialize_clusters(self):
        """Initialize network clusters using Louvain method"""
        # Detect communities using Louvain method
        clusters = community.best_partition(self.G)

        # Reorganize clusters into sets of nodes
        cluster_sets = defaultdict(set)
        for node, cluster_id in clusters.items():
            cluster_sets[cluster_id].add(node)
        self.clusters = dict(cluster_sets)

        # Identify bridge edges and internal edges
        for u, v in self.G.edges():
            u_cluster = clusters[u]
            v_cluster = clusters[v]

            if u_cluster == v_cluster:
                self.internal_edges[u_cluster].add((u, v))
            else:
                self.bridge_edges.add((u, v))

        print(f"Network partitioned into {len(self.clusters)} clusters")
        print(f"Identified {len(self.bridge_edges)} bridge edges")

    def _get_cluster_subgraph(self, cluster_id: int) -> nx.Graph:
        """Extract subgraph for a given cluster"""
        return self.G.subgraph(self.clusters[cluster_id]).copy()

    def _simulate_cluster(self, cluster_id: int) -> Tuple[np.ndarray, Dict]:
        """Run simulations for a single cluster"""
        subgraph = self._get_cluster_subgraph(cluster_id)
        n_nodes = subgraph.number_of_nodes()

        # Initialize arrays for results
        all_paths = np.zeros((self.params.cluster_simulations, n_nodes, n_nodes))
        node_list = list(subgraph.nodes())

        # Run simulations for this cluster
        for sim in range(self.params.cluster_simulations):
            # Generate infection times for internal edges
            infection_times = {
                (u, v): float(np.random.geometric(p=self.params.infection_prob))
                for u, v in subgraph.edges()
            }
            # Make symmetric
            infection_times.update({(v, u): t for (u, v), t in infection_times.items()})

            # Create weighted graph and compute paths
            weighted_G = nx.Graph()
            weighted_G.add_nodes_from(node_list)
            weighted_G.add_weighted_edges_from(
                [(u, v, infection_times[(u, v)]) for u, v in subgraph.edges()]
            )

            # Compute shortest paths within cluster
            for i, source in enumerate(node_list):
                lengths = nx.single_source_dijkstra_path_length(weighted_G, source)
                for j, target in enumerate(node_list):
                    all_paths[sim, i, j] = lengths.get(target, np.inf)

        # Calculate mean paths and create mapping
        mean_paths = np.mean(all_paths, axis=0)
        node_mapping = {old: new for new, old in enumerate(node_list)}

        return mean_paths, node_mapping

    def _cache_cluster_results(self):
        """Compute and cache results for all clusters"""
        print("Computing cluster-level results...")
        for cluster_id in tqdm(self.clusters.keys()):
            mean_paths, node_mapping = self._simulate_cluster(cluster_id)
            self.cluster_cache[cluster_id] = {
                'paths': mean_paths,
                'mapping': node_mapping,
                'reverse_mapping': {v: k for k, v in node_mapping.items()}
            }

    def _generate_potential_infection_times(self) -> Dict[Tuple[int, int], float]:
        """Generate infection times, using cached results where possible"""
        infection_times = {}

        # Generate times for bridge edges
        for u, v in self.bridge_edges:
            time = np.random.geometric(p=self.params.infection_prob)
            infection_times[(u, v)] = float(time)
            infection_times[(v, u)] = float(time)

        # Use cached results for internal edges where appropriate
        for cluster_id, edges in self.internal_edges.items():
            cache = self.cluster_cache[cluster_id]
            for u, v in edges:
                u_idx = cache['mapping'][u]
                v_idx = cache['mapping'][v]
                cached_time = cache['paths'][u_idx, v_idx]

                # Use cached time if it's below threshold
                if np.random.random() < self.params.cache_threshold:
                    infection_times[(u, v)] = cached_time
                    infection_times[(v, u)] = cached_time
                else:
                    # Generate new time
                    time = np.random.geometric(p=self.params.infection_prob)
                    infection_times[(u, v)] = float(time)
                    infection_times[(v, u)] = float(time)

        return infection_times

    def run_simulations(self):
        """Run simulations with clustering optimization"""
        # First, compute and cache cluster results
        self._cache_cluster_results()

        # Initialize arrays for full network results
        all_paths = []
        all_covering_times = []

        print("Running full network simulations...")
        for _ in tqdm(range(self.params.num_simulations)):
            # Generate infection times using cache where possible
            infection_times = self._generate_potential_infection_times()

            # Create weighted graph for full network
            weighted_G = nx.Graph()
            weighted_G.add_nodes_from(range(self.n_nodes))
            weighted_G.add_weighted_edges_from(
                [(u, v, infection_times[(u, v)]) for u, v in self.G.edges()]
            )

            # Compute paths for full network
            paths = np.full((self.n_nodes, self.n_nodes), np.inf)
            for i in range(self.n_nodes):
                lengths = nx.single_source_dijkstra_path_length(weighted_G, i)
                for j, length in lengths.items():
                    paths[i, j] = length

            # Compute covering times
            covering_times = np.max(paths, axis=1)

            all_paths.append(paths)
            all_covering_times.append(covering_times)

        # Average results across simulations
        self.infection_times = np.mean(all_paths, axis=0)
        self.covering_times = np.mean(all_covering_times, axis=0)

    def analyze_clustering_efficiency(self) -> Dict[str, Any]:
        """Analyze the effectiveness of the clustering approach"""
        analysis = {
            'num_clusters': len(self.clusters),
            'cluster_sizes': [len(nodes) for nodes in self.clusters.values()],
            'num_bridge_edges': len(self.bridge_edges),
            'cache_usage': self.params.cache_threshold,
            'potential_speedup': 1 - (len(self.bridge_edges) / self.G.number_of_edges())
        }
        return analysis

    # Inherit visualization methods from original implementation
    def get_node_rankings(self) -> pd.DataFrame:
        """Get node rankings based on mean infection covering times"""
        rankings = pd.DataFrame({
            'Node': [self.reverse_mapping[i] for i in range(self.n_nodes)],
            'MICT': self.covering_times,
            'Cluster': [next(cluster_id for cluster_id, nodes in self.clusters.items()
                             if i in nodes) for i in range(self.n_nodes)]
        })
        return rankings.sort_values('MICT')

    def plot_infection_times_heatmap(self):
        """Plot heatmap of mean infection times between nodes"""
        plt.figure(figsize=(10, 8))
        sns.heatmap(self.infection_times, cmap='inferno')
        plt.title('Mean Infection Times Between Nodes')
        plt.xlabel('Target Node')
        plt.ylabel('Source Node')
        plt.tight_layout()
        plt.show()

    def plot_covering_times_distribution(self):
        """Plot distribution of mean infection covering times"""
        plt.figure(figsize=(10, 6))
        sns.histplot(data=self.covering_times, bins=20)
        plt.title('Distribution of Mean Infection Covering Times')
        plt.xlabel('Time Steps')
        plt.ylabel('Count')
        plt.tight_layout()
        plt.show()

    def visualize_network_with_mict(self):
        """Visualize network with nodes colored by their MICT values"""
        plt.figure(figsize=(36, 24))

        # Create a new graph with original node labels
        G_orig = nx.relabel_nodes(self.G, self.reverse_mapping)
        pos = nx.spring_layout(G_orig)

        # Draw edges
        nx.draw_networkx_edges(G_orig, pos, alpha=0.2)

        # Draw nodes colored by MICT values
        node_colors = self.covering_times
        nodes = nx.draw_networkx_nodes(G_orig, pos, node_color=node_colors,
                                       node_size=500, cmap=plt.cm.YlOrRd)

        # Add node labels
        nx.draw_networkx_labels(G_orig, pos, font_size=7)

        # Add colorbar
        plt.colorbar(nodes, label='Mean Infection Covering Time')

        plt.title('Network Visualization with MICT Values')
        plt.axis('off')
        plt.tight_layout()
        plt.show()

    def visualize_network_with_clusters(self):
        """Visualize network with nodes colored by cluster and bridge edges highlighted"""
        plt.figure(figsize=(36, 24))

        # Create a new graph with original node labels
        G_orig = nx.relabel_nodes(self.G, self.reverse_mapping)
        pos = nx.spring_layout(G_orig, k=1 / np.sqrt(self.n_nodes))

        # Draw edges with different colors for bridges
        edge_colors = ['red' if tuple(sorted([self.node_mapping[u], self.node_mapping[v]])) in self.bridge_edges
                       else 'gray' for u, v in G_orig.edges()]
        nx.draw_networkx_edges(G_orig, pos, alpha=0.2, edge_color=edge_colors)

        # Draw nodes colored by cluster
        for cluster_id, nodes in self.clusters.items():
            nx.draw_networkx_nodes(G_orig, pos,
                                   nodelist=[self.reverse_mapping[n] for n in nodes],
                                   node_color=[cluster_id] * len(nodes),
                                   node_size=500,
                                   cmap=plt.cm.tab20)

        plt.title('Network Visualization with Clusters\n(Red edges are bridges)')
        plt.axis('off')
        plt.tight_layout()
        plt.show()


def main():
    """Example usage of the clustered epidemic model"""
    # Create a test network (using the same function as before)
    from article_impl import create_custom_network

    community_sizes = [(9, 9), (12, 12), (5, 5), (4, 4), (12, 12), (9,9), (15,15), (13,13), (14,14)]
    max_bridges = 3
    G = create_custom_network(community_sizes, max_bridges)

    # Set up model parameters
    params = ClusteredModelParameters(
        infection_prob=0.1,
        num_simulations=100,
        cluster_simulations=50,  # Run fewer simulations per cluster
        seed=23,
        cache_threshold=0.8
    )

    # Initialize and run model
    model = ClusteredEpidemicNetwork(G, params)
    model.run_simulations()

    # Analyze clustering efficiency
    efficiency_stats = model.analyze_clustering_efficiency()
    print("\nClustering Efficiency Analysis:")
    for key, value in efficiency_stats.items():
        print(f"{key}: {value}")

    # Generate visualizations
    model.plot_infection_times_heatmap()
    model.plot_covering_times_distribution()
    model.visualize_network_with_mict()
    model.visualize_network_with_clusters()

    # Print node rankings
    rankings = model.get_node_rankings()
    print("\nNode Rankings by Mean Infection Covering Time:")
    print(rankings)


if __name__ == "__main__":
    main()