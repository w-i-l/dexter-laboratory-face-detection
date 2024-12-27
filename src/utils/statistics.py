import os
from matplotlib import pyplot as plt
from utils.helper_functions import format_path

from clusters.k_means_cluster import KMeansAnalysis
from clusters.dbscan_cluster import DBSCANAnalysis
from clusters.hdbscan_cluster import HDBSCANAnalysis
from clusters.gmm_cluster import GMMAnalysis
import csv

class Statistics:
    def __init__(self, data_path: str):
        self._classes = ["dad", "mom", "dexter", "deedee", "unknown"]
        self._data_path = format_path(data_path)


    def get_attrs_for_class(self, class_name: str):
        class_path = os.path.join(self._data_path, class_name)
        class_path = format_path(class_path)

        if not os.path.exists(class_path):
            raise FileNotFoundError(f"Directory {class_path} not found")
        
        sizes = []
        aspect_ratios = []

        for file in os.listdir(class_path):
            if not file.endswith(".jpg"):
                continue

            file_path = os.path.join(class_path, file)
            file_path = format_path(file_path)

            image = plt.imread(file_path)
            sizes.append(image.shape[:-1])

            aspect_ratios.append(image.shape[1] / image.shape[0])

        return sizes, aspect_ratios
    

    def plot_class(self, class_name: str):
        sizes, aspect_ratios = self.get_attrs_for_class(class_name)
        aspect_ratios.sort()
        sizes.sort(key=lambda x: x[0] * x[1])
        
        # plot the sizes as points
        plt.scatter(*zip(*sizes))
        plt.title(f"Sizes for class {class_name}")
        plt.xlabel("Width")
        plt.ylabel("Height")
        
        # plot the aspect ratios as a line
        plt.figure()
        plt.plot(aspect_ratios, color="red")
        plt.title(f"Aspect ratios for class {class_name}")
        plt.xlabel("Index")
        plt.ylabel("Aspect ratio")

        
        plt.show()


if __name__ == "__main__":
    data_path = "../data/cropped_train"
    kmeans = KMeansAnalysis(data_path, n_clusters=70)
    gmm = GMMAnalysis(data_path, n_components=70)

    kmeans.plot_clusters()
    gmm.plot_clusters()

    # kmean_clusters = kmeans.get_cluster_centers()
    # kmean_clusters = list(zip(kmean_clusters[:, 0], kmean_clusters[:, 1]))
    # kmean_clusters.sort(key=lambda x: x[0] * x[1])
    # csv_path = "../data/clusters/kmeans_clusters.csv"
    # with open(csv_path, "w", newline="") as f:
    #     writer = csv.writer(f)
    #     writer.writerow(["Width", "Height"])
    #     writer.writerows(kmean_clusters)

    # gmm_clusters = gmm.get_cluster_centers()
    # gmm_clusters = list(zip(gmm_clusters[:, 0], gmm_clusters[:, 1]))
    # gmm_clusters.sort(key=lambda x: x[0] * x[1])
    # csv_path = "../data/clusters/gmm_clusters.csv"
    # with open(csv_path, "w", newline="") as f:
    #     writer = csv.writer(f)
    #     writer.writerow(["Width", "Height"])
    #     writer.writerows(gmm_clusters)

    # kmeans_clusters = "../data/clusters/kmeans_clusters.csv"
    # gmm_clusters = "../data/clusters/gmm_clusters.csv"

    # kmeans_clusters = list(csv.reader(open(kmeans_clusters)))
    # gmm_clusters = list(csv.reader(open(gmm_clusters)))

    # kmeans_clusters = [(float(row[0]), float(row[1])) for row in kmeans_clusters[1:]]
    # gmm_clusters = [(float(row[0]), float(row[1])) for row in gmm_clusters[1:]]

    # for kmean, gmm in zip(kmeans_clusters, gmm_clusters):
    #     print(kmean, gmm)
    #     print(abs(kmean[0] - gmm[0]), abs(kmean[1] - gmm[1]))
    #     print()
