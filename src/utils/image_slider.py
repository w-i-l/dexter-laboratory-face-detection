from utils.helper_functions import format_path
import csv
import numpy as np
from typing import Callable
import cv2
from face_detection.cnn_model import CNNModel
from utils.image_processing import ImageProcessing
from matplotlib import pyplot as plt
from tqdm import tqdm

class ImageSlider:
    def __init__(self, clusters_path: str):
        self.__initialize(clusters_path)


    def __initialize(self, clusters_path: str):
        self._clusters_path = format_path(clusters_path)
        self._clusters: list[tuple[float, float]] = []

        with open(self._clusters_path, 'r') as file:
            reader = csv.reader(file)
            next(reader) # Skip header

            for row in reader:
                x, y = row
                self._clusters.append((float(x), float(y)))


    def get_clusters(self) -> list[tuple[float, float]]:
        return self._clusters
    

    def slide_image(
        self,
        image: np.ndarray,
        callback: Callable[[np.ndarray], None]
    ):
        for cluster_size in tqdm(reversed(self._clusters), desc="Iterating over clusters"):
            cluster_width, cluster_height = cluster_size
            cluster_width = int(cluster_width)
            cluster_height = int(cluster_height)
            image_height, image_width, _ = image.shape

            # setting stride 10% of the cluster size
            stride_x = int(cluster_width * 0.2)
            stride_y = int(cluster_height * 0.2)

            for y in tqdm(range(0, int(image_height - cluster_height), stride_y), desc="Iterating over y"):
                for x in range(0, int(image_width - cluster_width), stride_x):
                    cropped_image = image[y:y+cluster_height, x:x+cluster_width]
                    # cv2.imshow("Cropped image", cropped_image)
                    # cv2.waitKey(0)
                    # cv2.destroyAllWindows()
                    callback(cropped_image)

model = CNNModel((ImageProcessing.IMAGE_WIDTH, ImageProcessing.IMAGE_HEIGHT, 3))
model.load_model("../models/face_detector.h5")

def show_image(image: np.ndarray):
    image = ImageProcessing.resize_image(image)
    print("Image shape", image.shape)
    prediction = model.predict(image)
    print("Prediction", prediction)

    if prediction >= 0.95:
        print("Face detected")
        plt.clf()
        plt.imshow(image)
        plt.title(f"Face Score: {prediction:.2f}")
        plt.draw()
        plt.pause(0.1)
    

if __name__ == "__main__":
    clusters_path = "../data/clusters/kmeans_clusters.csv"
    image_slider = ImageSlider(clusters_path)
    # clusters = image_slider.get_clusters()
    # print(clusters)

    image = ImageProcessing.read_image("../data/train/dexter/0322.jpg")
    print("Image shape", image.shape)

    plt.figure(figsize=(8, 6))
    plt.ion()  # Turn on interactive mode
    
    image_slider.slide_image(image, lambda x: show_image(x))

    plt.ioff()  # Turn off interactive mode when done
    plt.show()