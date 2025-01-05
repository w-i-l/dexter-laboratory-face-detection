from utils.helper_functions import format_path, compute_iou
import csv
import numpy as np
from typing import Callable
import cv2
from face_detection.cnn_model import CNNModel
from utils.image_processing import ImageProcessing
from matplotlib import pyplot as plt
from tqdm import tqdm
import os
import time
from utils.image_processing import ImageProcessing
from utils.image_slider import ImageSlider
from aggregators.cluster_aggregator import ClusterAggregator
from aggregators.nms_aggregator import NMSAggregator


class FacesAnnotations:
    def __init__(self, image_folder: str, annotation_path: str):
        self._annotation_path = format_path(annotation_path)
        self._image_folder = format_path(image_folder)

        self._annotations = {}
        self._read_annotations()

    def _read_annotations(self):
        with open(self._annotation_path, 'r') as file:
            for line in file:
                image_name, x_min, y_min, x_max, y_max, _ = line.split()
                box = (float(x_min), float(y_min), float(x_max), float(y_max))
                image_path = format_path(os.path.join(self._image_folder, image_name))

                if image_path not in self._annotations:
                    self._annotations[image_path] = [box]
                else:
                    self._annotations[image_path].append(box)

    def get_annotations(self, image_path: str) -> list[tuple[int, int, int, int]]:
        return self._annotations[image_path] if image_path in self._annotations else []
    

if __name__ == "__main__":
    # Load the model
    start_time = time.time()
    model = CNNModel((ImageProcessing.IMAGE_WIDTH, ImageProcessing.IMAGE_HEIGHT, 3))
    model.load_model("../models/face_detector.h5")
    print(f"Model loaded in {time.time() - start_time:.2f} seconds")

    # Load the clusters
    clusters_path = "../data/clusters/kmeans_clusters.csv"
    clusters = []
    with open(clusters_path, 'r') as file:
        reader = csv.reader(file)
        next(reader)  # Skip header
        for row in reader:
            x, y = row
            clusters.append((float(x), float(y)))
    print(f"\n Loaded {len(clusters)} clusters\n")

    image_slider = ImageSlider(clusters_path)

    class_name = "mom"
    dateset_type = "train"

    images_path = f"../data/{dateset_type}/{class_name}"
    annotations_path = f"../data/{dateset_type}/{class_name}_annotations.txt"
    faces_annotations = FacesAnnotations(images_path, annotations_path)

    try:
        prediction_file = open(f"../data/predictions/{class_name}_predictions.txt", "a+")
    except FileNotFoundError:
        os.makedirs("../data/predictions")
        prediction_file = open(f"../data/predictions/{class_name}_predictions.txt", "w+")
    
    if not os.path.exists(f"../data/found_bboxes/{class_name}"):
        os.makedirs(f"../data/found_bboxes/{class_name}")

    for image_name in tqdm(range(1, 1001)):
        image_name = str(image_name)
        image_name = image_name.zfill(4) # 0 padding up to 4 digits

        original_width = 480
        original_height = 360

        image_path = f"../data/{dateset_type}/{class_name}/{image_name}.jpg"
        image = cv2.imread(image_path)

        bboxes = []
        for bbox in faces_annotations.get_annotations(image_path):
            xmin, ymin, xmax, ymax = bbox
            xmin, ymin, xmax, ymax = int(xmin), int(ymin), int(xmax), int(ymax)
            bboxes.append((xmin, ymin, xmax, ymax))

        faces = image_slider.slide_image_batch(
            image,
            model, 
            batch_size=1024, 
            stride_factor=0.3, 
            face_threshold=0.8
        )

        # # Draw all detected faces
        # for face in faces:
        #     x, y, x2, y2 = face[0]
        #     cv2.rectangle(image, (x, y), (x2, y2), (0, 255, 0), 1)

        scores = [face[1] for face in faces]
        faces = [face[0] for face in faces]
        aggregated_faces = ClusterAggregator.get_face_boxes(faces, scores, iou_threshold=0.18)
        # aggregated_faces = NMSAggregator.get_face_boxes(faces, scores, iou_threshold=0.08) 

        # Draw aggregated faces
        for face,score in zip(aggregated_faces, scores):
            cv2.rectangle(image, (face[0], face[1]), (face[2], face[3]), (255, 255, 0), 2)
            prediction_file.write(f"{image_name}.jpg {face[0]} {face[1]} {face[2]} {face[3]} {score}\n")

        # Draw true faces
        for bbox in bboxes:
            cv2.rectangle(image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 0, 255), 2)

        image = cv2.resize(image, (original_width, original_height))

        # save the image
        save_path = f"../data/found_bboxes/{class_name}/{image_name}.jpg"
        cv2.imwrite(save_path, image)

    prediction_file.close()