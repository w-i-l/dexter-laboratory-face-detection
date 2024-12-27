from utils.helper_functions import format_path
import os
import numpy as np
import cv2
import csv
from typing import List, Tuple, Optional
from models.bounding_box import BoundingBox
from tqdm import tqdm

class PatchesExtractor:
    def __init__(
            self, 
            data_path: str, 
            clusters_path: str, 
            patch_overlap_threshold: float = 0.1, 
            face_overlap_threshold: float = 0.1,
            extractor_attempts_multiplier: int = 1000
        ):
        self._data_path = format_path(data_path)
        self._clusters_path = format_path(clusters_path)
        self._classes = ["dad", "mom", "dexter", "deedee", "unknown"]
        self._data = {}
        self._patch_overlap_threshold = patch_overlap_threshold
        self._face_overlap_threshold = face_overlap_threshold
        self._extractor_attempts_multiplier = extractor_attempts_multiplier
        self.__read_annotations()
        self.__read_clusters()

    def __read_clusters(self):
        self._clusters = []
        with open(self._clusters_path, 'r') as file:
            reader = csv.reader(file)
            next(reader)
            for row in reader:
                self._clusters.append((float(row[0]), float(row[1])))

    def __read_annotations(self):
        for class_name in self._classes[:-1]:
            annotation_path = format_path(os.path.join(self._data_path, f"{class_name}_annotations.txt"))
            with open(annotation_path, 'r') as file:
                for line in file:
                    image_name, x_min, y_min, x_max, y_max, _ = line.split()
                    box = BoundingBox(int(x_min), int(y_min), int(x_max), int(y_max))
                    image_path = format_path(os.path.join(self._data_path, class_name, image_name))
                    self._data.setdefault(image_path, []).append(box)

    def __get_valid_cluster_sizes(self, image_height: int, image_width: int) -> List[Tuple[int, int]]:
        return [(int(w), int(h)) for w, h in self._clusters 
                if w < image_width and h < image_height]

    def __is_valid_patch(self, patch: BoundingBox, existing_patches: List[BoundingBox], 
                        face_boxes: List[BoundingBox]) -> bool:
        # Check overlap with faces and inside faces
        for face_box in face_boxes:
            overlap = self.__calculate_overlap_percentage(patch, face_box)
            if overlap > self._face_overlap_threshold or self.__is_inside(patch, face_box):
                return False

        # Check overlap with other patches
        for existing_patch in existing_patches:
            overlap = self.__calculate_overlap_percentage(patch, existing_patch)
            if overlap > self._patch_overlap_threshold:
                return False

        return True

    def __calculate_overlap_percentage(self, box1: BoundingBox, box2: BoundingBox) -> float:
        # If boxes don't overlap, return 0
        if (box1.xmax <= box2.xmin or box2.xmax <= box1.xmin or 
            box1.ymax <= box2.ymin or box2.ymax <= box1.ymin):
            return 0.0

        # Calculate intersection area using max/min approach
        intersection_width = min(box1.xmax, box2.xmax) - max(box1.xmin, box2.xmin)
        intersection_height = min(box1.ymax, box2.ymax) - max(box1.ymin, box2.ymin)
        
        # Handle negative cases (no overlap)
        if intersection_width <= 0 or intersection_height <= 0:
            return 0.0
            
        intersection_area = intersection_width * intersection_height
        
        # Calculate areas
        area1 = box1.get_area()
        area2 = box2.get_area()
        
        # Use the smaller area for percentage calculation 
        # This ensures small boxes overlapping with large boxes are detected properly
        min_area = min(area1, area2)
        
        return intersection_area / min_area

    def __is_inside(self, box1: BoundingBox, box2: BoundingBox) -> bool:
        return (box1.xmin >= box2.xmin and box1.ymin >= box2.ymin and 
                box1.xmax <= box2.xmax and box1.ymax <= box2.ymax)

    def extract_patches_from_image(self, image_path: str, number_of_patches: int = 10) -> List[BoundingBox]:
        image = cv2.imread(image_path)
        height, width = image.shape[:2]
        valid_sizes = self.__get_valid_cluster_sizes(height, width)
        face_boxes = self._data.get(image_path, [])
        patches = []
        attempts = 0
        max_attempts = number_of_patches * self._extractor_attempts_multiplier  # More attempts to ensure we try enough times

        while len(patches) < number_of_patches and attempts < max_attempts:
            # Randomly select a size
            patch_width, patch_height = valid_sizes[np.random.randint(len(valid_sizes))]
            
            # Random position
            x = np.random.randint(0, width - patch_width)
            y = np.random.randint(0, height - patch_height)
            
            candidate_patch = BoundingBox(x, y, x + patch_width, y + patch_height)
            
            if self.__is_valid_patch(candidate_patch, patches, face_boxes):
                patches.append(candidate_patch)
            
            attempts += 1
            
        return patches

    def faces_in_image(self, image_path: str) -> List[BoundingBox]:
        return self._data.get(image_path, [])


if __name__ == "__main__":
    extractor = PatchesExtractor(
        "../data/train", 
        "../data/clusters/kmeans_clusters.csv",
        patch_overlap_threshold=0.2, 
        face_overlap_threshold=0
    )

    classes = ["dad", "mom", "dexter", "deedee"]
    for class_name in classes:
        class_path = os.path.join("../data/train", class_name)
        class_images = os.listdir(class_path)
        for image_name in tqdm(class_images, desc=f"Extracting patches for {class_name}"):
            image_path = os.path.join(class_path, image_name)

            patches = extractor.extract_patches_from_image(image_path, number_of_patches=10)
            image = cv2.imread(image_path)
            faces = extractor.faces_in_image(image_path)
            for index, patch in enumerate(patches):
                patch_image = image[patch.ymin:patch.ymax, patch.xmin:patch.xmax]
                save_path = os.path.join("../data/extracted_patches", f"{class_name}_{image_name}_{index}.jpg")
                cv2.imwrite(save_path, patch_image)
