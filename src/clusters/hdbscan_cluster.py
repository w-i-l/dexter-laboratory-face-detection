from typing import Tuple
import hdbscan
import numpy as np
import matplotlib.pyplot as plt
from clusters.base_cluster import BaseClusterAnalysis

class HDBSCANAnalysis(BaseClusterAnalysis):
    def __init__(self, data_path: str, min_cluster_size: int = 50):
        super().__init__(data_path)
        self.min_cluster_size = min_cluster_size
        self.clusterer = None

    def clusterize(self) -> Tuple[np.ndarray, None]:
        self.clusterer = hdbscan.HDBSCAN(
            min_cluster_size=self.min_cluster_size,
            min_samples=1,
            cluster_selection_epsilon=5.0
        )
        clusters = self.clusterer.fit_predict(self._X)
        return clusters, None

    def plot_clusters(self):
        if self._clusters is None:
            self._clusters, _ = self.clusterize()

        # Plot clustering results
        plt.figure(figsize=(15, 10))
        
        for char in self._classes:
            mask = self._labels == char
            plt.scatter(self._X[mask, 0], self._X[mask, 1], alpha=0.3, 
                       color=self._colors[char], label=f'{char} faces')
            
        plt.title(f'HDBSCAN Clustering Results\n(min_cluster_size={self.min_cluster_size})')
        plt.xlabel('Width (pixels)')
        plt.ylabel('Height (pixels)')
        plt.legend()
        plt.tight_layout()
        plt.show()
        
        # Plot membership probabilities
        plt.figure(figsize=(15, 10))
        plt.scatter(self._X[:, 0], self._X[:, 1], 
                   c=self.clusterer.probabilities_, cmap='viridis')
        plt.colorbar(label='Cluster Membership Probability')
        plt.title('HDBSCAN Cluster Membership Probabilities')
        plt.xlabel('Width (pixels)')
        plt.ylabel('Height (pixels)')
        plt.tight_layout()
        plt.show()

        self.print_cluster_statistics(self._clusters, None)