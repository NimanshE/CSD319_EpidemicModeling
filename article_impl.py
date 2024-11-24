import numpy as np
import networkx as nx
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
import warnings

warnings.filterwarnings('ignore')


@dataclass
class ModelParameters:
    """Parameters for the epidemic model"""
    infection_prob: float
    num_simulations: int
    seed: Optional[int] = None


class EpidemicNetworkModel:
    """Implementation of the paper's recommended epidemic modeling strategy"""

    def __init__(self, G: nx.Graph, params: ModelParameters):
        """
        Initialize the epidemic model with a network and parameters

        Args:
            G: NetworkX graph representing the contact network
            params: ModelParameters instance with simulation parameters
        """
        # Relabel nodes to integers for consistent processing
        self.node_mapping = {node: i for i, node in enumerate(G.nodes())}
        self.reverse_mapping = {i: node for node, i in self.node_mapping.items()}
        self.G = nx.relabel_nodes(G, self.node_mapping)

        self.params = params
        self.n_nodes = G.number_of_nodes()
        self.infection_times = None
        self.covering_times = None

        if params.seed is not None:
            np.random.seed(params.seed)

    def _generate_potential_infection_times(self) -> Dict[Tuple[int, int], float]:
        """Generate geometric distributed infection times for each edge"""
        infection_times = {}
        for u, v in self.G.edges():
            # Geometric distribution for infection times as per paper
            time = np.random.geometric(p=self.params.infection_prob)
            infection_times[(u, v)] = float(time)  # Convert to float for NetworkX compatibility
            infection_times[(v, u)] = float(time)  # Undirected graph
        return infection_times

    def _compute_shortest_paths(self, infection_times: Dict[Tuple[int, int], float]) -> np.ndarray:
        """Compute all-pairs shortest paths with infection times as weights"""
        # Create weighted graph using infection times
        weighted_G = nx.Graph()
        weighted_G.add_nodes_from(range(self.n_nodes))
        weighted_G.add_weighted_edges_from(
            [(u, v, infection_times[(u, v)]) for u, v in self.G.edges()]
        )

        # Compute shortest paths matrix
        paths = np.full((self.n_nodes, self.n_nodes), np.inf)
        for i in range(self.n_nodes):
            lengths = nx.single_source_dijkstra_path_length(weighted_G, i)
            for j, length in lengths.items():
                paths[i, j] = length
        return paths

    def _single_simulation(self) -> Tuple[np.ndarray, np.ndarray]:
        """Run a single simulation of the epidemic spread"""
        infection_times = self._generate_potential_infection_times()
        paths = self._compute_shortest_paths(infection_times)

        # Compute covering times for each starting node
        covering_times = np.max(paths, axis=1)

        return paths, covering_times

    def run_simulations(self):
        """Run multiple simulations and aggregate results"""
        all_paths = []
        all_covering_times = []

        # Use sequential processing for small number of simulations
        for _ in tqdm(range(self.params.num_simulations)):
            paths, covering_times = self._single_simulation()
            all_paths.append(paths)
            all_covering_times.append(covering_times)

        # Average results across simulations
        self.infection_times = np.mean(all_paths, axis=0)
        self.covering_times = np.mean(all_covering_times, axis=0)

    def get_node_rankings(self) -> pd.DataFrame:
        """Get node rankings based on mean infection covering times"""
        rankings = pd.DataFrame({
            'Node': [self.reverse_mapping[i] for i in range(self.n_nodes)],
            'MICT': self.covering_times
        })
        return rankings.sort_values('MICT')

    def plot_infection_times_heatmap(self):
        """Plot heatmap of mean infection times between nodes"""
        plt.figure(figsize=(10, 8))
        sns.heatmap(self.infection_times, cmap='YlOrRd')
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
        plt.figure(figsize=(12, 8))

        # Create a new graph with original node labels
        G_orig = nx.relabel_nodes(self.G, self.reverse_mapping)
        pos = nx.spring_layout(G_orig)

        # Draw edges
        nx.draw_networkx_edges(G_orig, pos, alpha=0.2)

        # Draw nodes
        node_colors = self.covering_times
        nodes = nx.draw_networkx_nodes(G_orig, pos, node_color=node_colors,
                                       node_size=500, cmap=plt.cm.YlOrRd)

        # Add node labels
        nx.draw_networkx_labels(G_orig, pos, font_size=8)

        # Add colorbar
        plt.colorbar(nodes, label='Mean Infection Covering Time')

        plt.title('Network Visualization with MICT Values')
        plt.axis('off')
        plt.tight_layout()
        plt.show()


def create_test_network() -> nx.Graph:
    """Create a test network similar to examples in the paper"""
    # Create two connected communities
    G1 = nx.grid_2d_graph(3, 3)  # 3x3 grid for first community
    G2 = nx.grid_2d_graph(3, 3)  # 3x3 grid for second community

    # Relabel nodes to avoid conflicts
    G2 = nx.relabel_nodes(G2, {node: (node[0] + 3, node[1] + 3) for node in G2.nodes()})

    # Combine communities
    G = nx.union(G1, G2)

    # Add bridges between communities
    G.add_edge((2, 2), (3, 3))  # Bridge 1
    G.add_edge((2, 1), (3, 2))  # Bridge 2

    return G

def create_complex_test_network() -> nx.Graph:
    """Create a more complex test network with 4 communities and multiple inter-community bridges"""
    # Create four connected communities
    G1 = nx.grid_2d_graph(3, 3)  # 3x3 grid for first community
    G2 = nx.grid_2d_graph(3, 3)  # 3x3 grid for second community
    G3 = nx.grid_2d_graph(3, 3)  # 3x3 grid for third community
    G4 = nx.grid_2d_graph(3, 3)  # 3x3 grid for fourth community

    # Relabel nodes to avoid conflicts
    G2 = nx.relabel_nodes(G2, {node: (node[0] + 3, node[1] + 3) for node in G2.nodes()})
    G3 = nx.relabel_nodes(G3, {node: (node[0] + 6, node[1]) for node in G3.nodes()})
    G4 = nx.relabel_nodes(G4, {node: (node[0] + 6, node[1] + 3) for node in G4.nodes()})

    # Combine communities
    G = nx.union(G1, G2)
    G = nx.union(G, G3)
    G = nx.union(G, G4)

    # Add bridges between communities
    G.add_edge((2, 2), (3, 3))  # Bridge between G1 and G2
    G.add_edge((2, 1), (3, 2))  # Bridge between G1 and G2
    G.add_edge((2, 0), (6, 0))  # Bridge between G1 and G3
    G.add_edge((2, 2), (6, 2))  # Bridge between G1 and G3
    G.add_edge((5, 3), (6, 3))  # Bridge between G2 and G4
    G.add_edge((5, 5), (6, 5))  # Bridge between G2 and G4
    G.add_edge((8, 0), (8, 3))  # Bridge between G3 and G4
    G.add_edge((7, 2), (7, 5))  # Bridge between G3 and G4

    return G


# Example usage and demonstration
def main():
    # Create test network
    G = create_complex_test_network()

    # Set up model parameters
    params = ModelParameters(
        infection_prob=0.1,
        num_simulations=100,
        seed=18
    )

    # Initialize and run model
    model = EpidemicNetworkModel(G, params)
    model.run_simulations()

    # Generate visualizations
    model.plot_infection_times_heatmap()
    model.plot_covering_times_distribution()
    model.visualize_network_with_mict()

    # Print node rankings
    rankings = model.get_node_rankings()
    print("\nNode Rankings by Mean Infection Covering Time:")
    print(rankings)


if __name__ == "__main__":
    main()