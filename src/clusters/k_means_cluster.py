from typing import Tuple
from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt
from clusters.base_cluster import BaseClusterAnalysis

class KMeansAnalysis(BaseClusterAnalysis):
    def __init__(self, data_path: str, n_clusters: int = 3):
        super().__init__(data_path)
        self.n_clusters = n_clusters
        self.kmeans = None

    def clusterize(self) -> Tuple[np.ndarray, np.ndarray]:
        self.kmeans = KMeans(n_clusters=self.n_clusters, random_state=42)
        clusters = self.kmeans.fit_predict(self._X)
        centers = self.kmeans.cluster_centers_
        return clusters, centers

    def get_cluster_centers(self) -> np.ndarray:
        if self.kmeans is None:
            self._clusters, centers = self.clusterize()
        return self.kmeans.cluster_centers_

    def plot_clusters(self):
        if self._clusters is None:
            self._clusters, centers = self.clusterize()

        # Calculate data boundaries with padding
        x_min, x_max = self._X[:, 0].min(), self._X[:, 0].max()
        y_min, y_max = self._X[:, 1].min(), self._X[:, 1].max()
        x_padding = (x_max - x_min) * 0.1
        y_padding = (y_max - y_min) * 0.1
        x_min -= x_padding
        x_max += x_padding
        y_min -= y_padding
        y_max += y_padding

        # Create figure with adjusted size and layout
        fig = plt.figure(figsize=(15, 10))
        # Add more space on the right for colorbar
        gs = plt.GridSpec(1, 20)
        ax = plt.subplot(gs[0, :19])  # Main plot takes 19/20 of the width
        
        # Create a colormap
        cmap = plt.cm.get_cmap('tab20')  # Using tab20 for distinct colors up to 20 clusters
        
        # Plot points colored by cluster
        scatter = ax.scatter(self._X[:, 0], self._X[:, 1], 
                         c=self._clusters, 
                         cmap=cmap,
                         alpha=0.6,
                         label='Data points')
        
        # Plot cluster centers
        # ax.scatter(self.kmeans.cluster_centers_[:, 0], 
        #         self.kmeans.cluster_centers_[:, 1], 
        #         c='red', 
        #         marker='x', 
        #         s=200, 
        #         linewidth=3, 
        #         label='Cluster centers')
        
        # Plot circles around centers to show cluster regions
        for i, center in enumerate(self.kmeans.cluster_centers_):
            circle = plt.Circle(center, 
                           self.kmeans.inertia_**0.5/len(self._X)**0.5, 
                           fill=False, 
                           color='red', 
                           linestyle='--', 
                           alpha=0.5)
            # ax.add_patch(circle)
            # Add cluster center labels
            ax.text(center[0], center[1], f'C{i}',
                   horizontalalignment='center',
                   verticalalignment='center',
                   color='white',
                   fontweight='bold')
        
        # Add colorbar in the extra space
        cbar_ax = plt.subplot(gs[0, 19])  # Colorbar takes position 19
        plt.colorbar(scatter, cax=cbar_ax, label='Cluster ID')
        
        ax.set_title(f'K-means Clustering Analysis\n(n_clusters={self.n_clusters})')
        ax.set_xlabel('Width (pixels)')
        ax.set_ylabel('Height (pixels)')
        
        # Set axis limits with padding
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        
        # Add legend
        ax.legend(loc='upper left')
        
        plt.tight_layout()
        plt.show()

        # Print statistics about clusters
        self.print_cluster_statistics(self._clusters, self.kmeans.cluster_centers_)

        # Plot inertia (within-cluster sum of squares) per cluster
        plt.figure(figsize=(15, 5))
        cluster_inertias = []
        for i in range(self.n_clusters):
            cluster_points = self._X[self._clusters == i]
            center = self.kmeans.cluster_centers_[i]
            inertia = np.sum(np.linalg.norm(cluster_points - center, axis=1)**2)
            cluster_inertias.append(inertia)

        plt.bar(range(self.n_clusters), cluster_inertias)
        plt.title('Within-Cluster Sum of Squares per Cluster')
        plt.xlabel('Cluster ID')
        plt.ylabel('Inertia')
        plt.xticks(range(self.n_clusters))
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()