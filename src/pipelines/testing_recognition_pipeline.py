from utils.helper_functions import format_path, compute_iou
import csv
import numpy as np
from typing import Callable, List, Tuple, Dict
import cv2
from face_recognition.cnn_model import CNNModel
from utils.image_processing import ImageProcessing
from matplotlib import pyplot as plt
from tqdm import tqdm
import os
import time
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


    def get_annotations(self, image_path: str) -> list[tuple[tuple[int, int, int, int], str]]:
        return self._annotations[image_path] if image_path in self._annotations else []



def process_images_in_batches(model: CNNModel, 
                            annotations: FacesAnnotations, 
                            batch_size: int = 1024) -> Tuple[List[Tuple[str, float]], Dict[str, List[Tuple[str, float, tuple]]]]:
    """
    Process images in batches and return predictions along with image-specific data
    Returns:
        Tuple containing:
        - List of (predicted_class, confidence) for all faces
        - Dict mapping image paths to list of (predicted_class, confidence, bbox)
    """
    batch = []
    batch_metadata = []  # Store (image_path, bbox) for each face
    all_predictions = []
    image_predictions = {}

    # Process each image
    for image_path in tqdm(annotations._annotations.keys(), desc="Processing images"):
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"Image not found: {image_path}")

        faces = annotations.get_annotations(image_path)
        image_predictions[image_path] = []
        
        for bbox in faces:
            x_min, y_min, x_max, y_max = map(int, bbox)
            face = image[y_min:y_max, x_min:x_max]
            face = ImageProcessing.resize_image(face)
            
            batch.append(face)
            batch_metadata.append((image_path, bbox))
            
            # Process batch when it reaches batch_size
            if len(batch) >= batch_size:
                predictions = model.predict_batch(np.array(batch))
                
                # Store predictions
                for (img_path, box), (pred_class, conf) in zip(batch_metadata, predictions):
                    all_predictions.append((pred_class, conf))
                    image_predictions[img_path].append((pred_class, conf, box))
                
                # Clear batch
                batch = []
                batch_metadata = []

    # Process remaining images in last batch
    if batch:
        predictions = model.predict_batch(np.array(batch))
        for (img_path, box), (pred_class, conf) in zip(batch_metadata, predictions):
            all_predictions.append((pred_class, conf))
            image_predictions[img_path].append((pred_class, conf, box))

    return all_predictions, image_predictions

def save_annotated_images(image_predictions: Dict[str, List[Tuple[str, float, tuple]]], 
                         output_dir: str,
                         colors: Dict[str, tuple]):
    """Save images with annotated predictions"""
    os.makedirs(output_dir, exist_ok=True)
    
    for image_path, predictions in tqdm(image_predictions.items(), desc="Saving annotated images"):
        image = cv2.imread(image_path)
        
        # Draw each prediction
        for pred_class, confidence, bbox in predictions:
            x_min, y_min, x_max, y_max = map(int, bbox)
            color = colors[pred_class]
            
            # Draw box
            cv2.rectangle(image, (x_min, y_min), (x_max, y_max), color, 2)
            
            # Add label with confidence
            label = f"{pred_class}: {confidence:.2f}"
            cv2.putText(image, label, (x_min, y_min-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,100,255), 2)
        
        # Save annotated image
        output_path = os.path.join(output_dir, os.path.basename(image_path))
        cv2.imwrite(output_path, image)

if __name__ == "__main__":
    # Define colors for visualization
    colors = {
        "dad": (164, 235, 52),
        "mom": (222, 82, 138),
        "deedee": (82, 199, 222),
        "dexter": (222, 126, 82),
        "unknown": (66, 105, 158)
    }

    # Load model
    start_time = time.time()
    model = CNNModel((ImageProcessing.IMAGE_WIDTH, ImageProcessing.IMAGE_HEIGHT, 3))
    model.load_model("../models/face_recognizer.h5")
    print(f"Model loaded in {time.time() - start_time:.2f} seconds")

    # Process each class
    output_base_dir = "../data/face_recognition"

    Class_name = "all"
    print(f"\nProcessing class: {Class_name}")

    annotaion_path = "../data/solutions/ground_truth/task2/all_annotations.txt"
    annotations = []
    if not os.path.exists(annotaion_path):
        for class_name in ["dad", "mom", "dexter", "deedee"]:
            with open (f"../data/solutions/ground_truth/task2/task2_{class_name}_gt_validare.txt", "r") as f:
                for line in f:
                    image_name, x_min, y_min, x_max, y_max = line.strip().split()
                    annotations.append((image_name, int(x_min), int(y_min), int(x_max), int(y_max), class_name))

        with open(annotaion_path, "w+") as f:
            annotations = sorted(annotations, key=lambda x: x[0])
            for annotation in annotations:
                f.write(" ".join(map(str, annotation)) + "\n")
            

    # Load annotations
    annotations = FacesAnnotations(
        f"../data/validation/{Class_name}", 
        f"../data/predictions/{Class_name}_predictions.txt"
    )

    # Process images
    all_predictions, image_predictions = process_images_in_batches(
        model, 
        annotations, 
        batch_size=1024
    )

    output = { class_name: {
        "detections": [],
        "scores": [],
        "file_names": []
    } for class_name in ["dad", "mom", "dexter", "deedee"]}

    for image_path, predictions in image_predictions.items():
        for pred_class, confidence, bbox in predictions:
            if pred_class not in output:
                continue
            
            output[pred_class]["detections"].append(bbox)
            output[pred_class]["scores"].append(confidence)
            output[pred_class]["file_names"].append(os.path.basename(image_path))

    for class_name in ["dad", "mom", "dexter", "deedee"]:
        output_path = f"../data/solutions/task2/"
        if not os.path.exists(output_path):
            os.makedirs(output_path)

        detections = np.array(output[class_name]["detections"])
        scores = np.array(output[class_name]["scores"])
        file_names = np.array(output[class_name]["file_names"])

        np.save(output_path + f"detections_{class_name}.npy", detections)
        np.save(output_path + f"scores_{class_name}.npy", scores)
        np.save(output_path + f"file_names_{class_name}.npy", file_names)

