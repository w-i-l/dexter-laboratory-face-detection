from typing import Tuple
from sklearn.mixture import GaussianMixture
import numpy as np
import matplotlib.pyplot as plt
from clusters.base_cluster import BaseClusterAnalysis
from matplotlib.patches import Ellipse

class GMMAnalysis(BaseClusterAnalysis):
    def __init__(self, data_path: str, n_components: int = 5):
        super().__init__(data_path)
        self.n_components = n_components
        self.gmm = None

    def clusterize(self) -> Tuple[np.ndarray, np.ndarray]:
        self.gmm = GaussianMixture(n_components=self.n_components, 
                                 covariance_type='full')
        self.gmm.fit(self._X)
        clusters = self.gmm.predict(self._X)
        return clusters, self.gmm.means_
    
    def get_cluster_centers(self) -> np.ndarray:
        if self.gmm is None:
            self._clusters, centers = self.clusterize()
        return self.gmm.means_

    def plot_clusters(self):
        if self._clusters is None:
            self._clusters, _ = self.clusterize()

        # Calculate data boundaries with padding
        x_min, x_max = self._X[:, 0].min(), self._X[:, 0].max()
        y_min, y_max = self._X[:, 1].min(), self._X[:, 1].max()
        x_padding = (x_max - x_min) * 0.1
        y_padding = (y_max - y_min) * 0.1
        x_min -= x_padding
        x_max += x_padding
        y_min -= y_padding
        y_max += y_padding

        # Create visualization grid
        x = np.linspace(x_min, x_max, 100)
        y = np.linspace(y_min, y_max, 100)
        X_grid, Y_grid = np.meshgrid(x, y)
        XX = np.array([X_grid.ravel(), Y_grid.ravel()]).T
        
        # Get log probability and reshape
        Z = self.gmm.score_samples(XX).reshape(X_grid.shape)
        
        # Create figure with adjusted size and layout
        fig = plt.figure(figsize=(15, 10))
        # Add more space on the right for colorbars
        gs = plt.GridSpec(1, 20)
        ax = plt.subplot(gs[0, :18])  # Main plot takes 18/20 of the width
        
        # Plot points colored by cluster assignment
        scatter = ax.scatter(self._X[:, 0], self._X[:, 1], 
                         c=self._clusters,
                         cmap='tab20',
                         alpha=0.6)
        
        # Add colorbar for cluster assignments in the extra space
        cbar_ax1 = plt.subplot(gs[0, 18])  # First colorbar takes position 18
        plt.colorbar(scatter, cax=cbar_ax1, label='Cluster ID')
        
        # Plot density contours
        contour = ax.contour(X_grid, Y_grid, Z, levels=15, 
                         cmap='viridis', alpha=0.5)
        cbar_ax2 = plt.subplot(gs[0, 19])  # Second colorbar takes position 19
        plt.colorbar(contour, cax=cbar_ax2, label='Log Probability Density')
        
        # Plot cluster centers
        centers = self.gmm.means_
        # ax.scatter(centers[:, 0], centers[:, 1], 
        #         c='red', marker='x', s=200, linewidth=3, 
        #         label='Gaussian centers')
        
        # Plot covariance ellipses for each Gaussian component
        for i, (mean, covar) in enumerate(zip(self.gmm.means_, 
                                            self.gmm.covariances_)):
            v, w = np.linalg.eigh(covar)
            v = 2. * np.sqrt(2.) * np.sqrt(v)
            u = w[0] / np.linalg.norm(w[0])
            angle = np.arctan(u[1] / u[0])
            angle = 180. * angle / np.pi
            
            # Plot 1 and 2 standard deviation contours
            for n_std in [1, 2]:
                ell = Ellipse(xy=mean, 
                            width=v[0] * n_std, 
                            height=v[1] * n_std,
                            angle=180. + angle,
                            edgecolor='black',
                            facecolor='none',
                            linestyle='--',
                            alpha=0.3)
                # ax.add_patch(ell)
                if n_std == 2:
                    ax.text(mean[0], mean[1], f'G{i}',
                           horizontalalignment='center',
                           verticalalignment='center')
        
        ax.set_title(f'GMM Clustering Results\n' + 
                    f'n_components={self.n_components}, ' +
                    f'covariance_type={self.gmm.covariance_type}')
        ax.set_xlabel('Width (pixels)')
        ax.set_ylabel('Height (pixels)')
        
        # Set axis limits with padding
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        
        # Add legend
        ax.legend(['Data points', 'Gaussian centers', 
                  '1σ contour', '2σ contour'])
        
        plt.tight_layout()
        plt.show()

        # Print cluster statistics
        self.print_cluster_statistics(self._clusters, self.gmm.means_)

        # Plot cluster responsibilities
        plt.figure(figsize=(15, 10))
        probs = self.gmm.predict_proba(self._X)
        
        # Create a grid of subplots for component probabilities
        n_cols = min(3, self.n_components)
        n_rows = (self.n_components + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols, 
                                figsize=(15, 5*n_rows))
        if n_rows == 1:
            axes = axes.reshape(1, -1)
            
        for i, ax in enumerate(axes.flat):
            if i < self.n_components:
                scatter = ax.scatter(self._X[:, 0], self._X[:, 1],
                                   c=probs[:, i],
                                   cmap='viridis',
                                   alpha=0.6)
                ax.set_title(f'Component {i} Responsibilities')
                # Set consistent axis limits for all subplots
                ax.set_xlim(x_min, x_max)
                ax.set_ylim(y_min, y_max)
                plt.colorbar(scatter, ax=ax)
            else:
                ax.axis('off')
                
        plt.tight_layout()
        plt.show()