from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple
from matplotlib import pyplot as plt
import numpy as np
import os

class BaseClusterAnalysis(ABC):
    def __init__(self, data_path: str):
        """Initialize base clustering analysis class
        
        Args:
            data_path (str): Path to directory containing face images
        """
        self._classes = ["dad", "mom", "dexter", "deedee", "unknown"]
        self._data_path = data_path
        self._colors = {
            'dexter': 'blue',
            'deedee': 'yellow',
            'mom': 'purple',
            'dad': 'green',
            'unknown': 'red'
        }
        self._X = None  # Feature matrix
        self._labels = None  # Original character labels
        self._clusters = None  # Cluster assignments
        self._prepare_data()

    def _prepare_data(self):
        """Collect and prepare data for clustering"""
        all_widths, all_heights, labels = [], [], []
        for char in self._classes:
            stats = self.get_face_statistics(char)
            all_widths.extend(stats['widths'])
            all_heights.extend(stats['heights'])
            labels.extend([char] * len(stats['widths']))
            
        self._X = np.array(list(zip(all_widths, all_heights)))
        self._labels = np.array(labels)

    def get_face_statistics(self, class_name: str) -> Dict[str, np.ndarray]:
        """Get statistics from face images in class folder"""
        class_path = os.path.join(self._data_path, class_name)
        
        widths, heights, aspect_ratios = [], [], []
        
        for file in os.listdir(class_path):
            if not file.endswith(('.jpg', '.png')):
                continue
                
            image = plt.imread(os.path.join(class_path, file))
            height, width = image.shape[:2]
            
            widths.append(width)
            heights.append(height)
            aspect_ratios.append(width/height)
            
        return {
            'widths': np.array(widths),
            'heights': np.array(heights),
            'aspect_ratios': np.array(aspect_ratios)
        }

    @abstractmethod
    def clusterize(self) -> Tuple[np.ndarray, Optional[List]]:
        """Perform clustering and return cluster assignments"""
        pass

    @abstractmethod
    def plot_clusters(self):
        """Plot clustering results"""
        pass

    @abstractmethod
    def get_cluster_centers(self) -> np.ndarray:
        """Get cluster centers"""
        pass

    def print_cluster_statistics(self, clusters: np.ndarray, centers: np.ndarray):
        """Print statistics for each cluster"""
        print("\nCluster Statistics:")
        for i in range(len(np.unique(clusters))):
            cluster_mask = clusters == i
            cluster_chars = self._labels[cluster_mask]
            
            if centers is not None:
                print(f"\nCluster {i+1}:")
                print(f"Center: Width={centers[i][0]:.1f}, Height={centers[i][1]:.1f}")
            
            print("Character composition:")
            for char in self._classes:
                count = np.sum(cluster_chars == char)
                if len(cluster_chars) > 0:
                    percentage = (count / len(cluster_chars)) * 100
                    print(f"  {char}: {count} faces ({percentage:.1f}%)")