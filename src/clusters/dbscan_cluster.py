from typing import Optional, Tuple
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
import numpy as np
import matplotlib.pyplot as plt
from clusters.base_cluster import BaseClusterAnalysis

class DBSCANAnalysis(BaseClusterAnalysis):
    def __init__(self, data_path: str, eps: Optional[float] = None, min_samples: int = 5):
        super().__init__(data_path)
        self.eps = eps
        self.min_samples = min_samples
        self.dbscan = None

    def _estimate_eps(self) -> float:
        """Estimate eps parameter if not provided"""
        neigh = NearestNeighbors(n_neighbors=2)
        neigh.fit(self._X)
        distances, _ = neigh.kneighbors(self._X)
        distances = np.sort(distances[:, 1])
        
        plt.figure(figsize=(10, 6))
        plt.plot(distances)
        plt.title('K-distance Graph')
        plt.xlabel('Points')
        plt.ylabel('Distance to nearest neighbor')
        plt.show()
        
        return np.percentile(distances, 90)

    def clusterize(self) -> Tuple[np.ndarray, None]:
        if self.eps is None:
            self.eps = self._estimate_eps()
            print(f"Estimated eps: {self.eps}")

        self.dbscan = DBSCAN(eps=self.eps, min_samples=self.min_samples)
        clusters = self.dbscan.fit_predict(self._X)
        return clusters, None

    def plot_clusters(self):
        if self._clusters is None:
            self._clusters, _ = self.clusterize()

        unique_clusters = np.unique(self._clusters)
        colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_clusters)))
        
        plt.figure(figsize=(12, 8))
        
        for cluster_id, color in zip(unique_clusters, colors):
            mask = self._clusters == cluster_id
            if cluster_id == -1:
                plt.scatter(self._X[mask, 0], self._X[mask, 1], 
                          c='black', alpha=0.1, label='Noise')
            else:
                plt.scatter(self._X[mask, 0], self._X[mask, 1], 
                          c=[color], alpha=0.5, label=f'Cluster {cluster_id+1}')
        
        n_clusters = len(unique_clusters) - (1 if -1 in unique_clusters else 0)
        n_noise = list(self._clusters).count(-1)
        plt.title(f'DBSCAN Clustering Results\n' + 
                 f'eps={self.eps:.2f}, min_samples={self.min_samples}\n' +
                 f'clusters: {n_clusters}, noise points: {n_noise}')
        plt.xlabel('Width (pixels)')
        plt.ylabel('Height (pixels)')
        plt.legend()
        plt.tight_layout()
        plt.show()

        self.print_cluster_statistics(self._clusters, None)


    def get_cluster_centers(self):
        if self.dbscan is None:
            self.clusterize()
        
        return self.dbscan.components_