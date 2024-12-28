from typing import List
from models.image_annotation import ImageAnnotation
from utils.image_processing import ImageProcessing
from utils.helper_functions import format_path
import os
from tqdm import tqdm

class DataSet:
    def __init__(self, faces_path: str, non_faces_path: str):
        self._faces_path = format_path(faces_path)
        self._non_faces_path = format_path(non_faces_path)
        self.classes = ["dad", "mom", "dexter", "deedee", "unknown"]
        self.labels = ["non_face", "face"]

    
    def read_data(self, read_non_faces: bool = True) -> tuple[List, List]:
        data, labels = [], []

        # reading faces
        data_faces, labels_faces = self.read_faces()
        data.extend(data_faces)
        labels.extend(labels_faces)

        if not read_non_faces:
            return data, labels
        
        # reading non-faces
        data_non_faces, labels_non_faces = self.read_non_faces()
        data.extend(data_non_faces)
        labels.extend(labels_non_faces)

        return data, labels
    

    def read_faces(self) -> tuple[List, List]:
        data, labels = [], []

        for class_name in self.classes:
            class_path = os.path.join(self._faces_path, class_name)
            class_path = format_path(class_path)

            if not os.path.exists(class_path):
                raise FileNotFoundError(f"Directory {class_path} not found")
            
            for file in tqdm(os.listdir(class_path), desc=f"Reading {class_name}"):
                if not file.endswith(".jpg"):
                    continue

                file_path = os.path.join(class_path, file)
                file_path = format_path(file_path)

                image = ImageProcessing.read_image(file_path)
                data.append(image)
                labels.append(1)

        return data, labels
    

    def read_non_faces(self, size: int = None) -> tuple[List, List]:
        data, labels = [], []

        class_path = self._non_faces_path
        for file in tqdm(os.listdir(class_path), desc="Reading non-faces"):
            if not file.endswith(".jpg"):
                continue

            file_path = os.path.join(class_path, file)
            file_path = format_path(file_path)

            image = ImageProcessing.read_image(file_path)
            data.append(image)
            labels.append(0)

            if size is not None and len(data) >= size:
                break

        return data, labels
        


    def split_dataset(self, data, labels, split_factor: float = 0.8) -> tuple[list, list, list, list]:
        _data = { label: [] for label in range(len(self.labels)) }

        for image, label in tqdm(zip(data, labels), desc="Splitting dataset"):
            _data[label].append(image)
        train_data, test_data = [], []
        train_labels, test_labels = [], []

        for label in range(len(self.labels)):
            split_index = int(len(_data[label]) * split_factor)
            train_data.extend(_data[label][:split_index])
            test_data.extend(_data[label][split_index:])
            train_labels.extend([label] * split_index)
            test_labels.extend([label] * (len(_data[label]) - split_index))

        return train_data, test_data, train_labels, test_labels
        

if __name__ == "__main__":
    faces_path = "../data/cropped_train"
    non_faces_path = "../data/extracted_patches"
    dataset = DataSet(faces_path, non_faces_path)
    data, labels = dataset.read_data()
    print(data[0].shape)
    print(f"Data: {len(data)}")
    print(f"Labels: {len(labels)}")
    train_data, test_data, train_labels, test_labels = dataset.split_dataset(data, labels)
    print(f"Train data: {len(train_data)}")
    print(f"Test data: {len(test_data)}")
    print(f"Train labels: {len(train_labels)}")
    print(f"Test labels: {len(test_labels)}")