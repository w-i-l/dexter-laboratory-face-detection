from utils.helper_functions import format_path
import os
from models.bounding_box import BoundingBox
from models.image_annotation import ImageAnnotation
from utils.image_processing import ImageProcessing
import cv2
from tqdm import tqdm

class DataOrganizer:
    def __init__(self, data_path: str, destination_path: str):
        self._classes = ["dad", "mom", "dexter", "deedee", "unknown"]
        self._data_path = format_path(data_path)
        self._destination_path = format_path(destination_path)
        self._data: dict[str, list[ImageAnnotation]] = {class_name: [] for class_name in self._classes}


    def organize_data(self):
        self.__create_folders()

        for class_name in self._classes[:-1]:
            self.__organize_class(class_name)

        for class_name in self._classes:
            self.__write_class(class_name)


    def __format_image_name(self, index: int) -> str:
        return f"{index:04d}"
    

    def __create_folders(self):
        for class_name in self._classes:
            destination_path = os.path.join(self._destination_path, class_name)
            destination_path = format_path(destination_path)

            if not os.path.exists(destination_path):
                os.makedirs(destination_path)
            else:
                for file in os.listdir(destination_path):
                    file_path = os.path.join(destination_path, file)
                    file_path = format_path(file_path)
                    os.remove(file_path)


    def __write_class(self, class_name: str):
        destination_path = os.path.join(self._destination_path, class_name)
        destination_path = format_path(destination_path)

        if not os.path.exists(destination_path):    
            raise FileNotFoundError(f"Directory {destination_path} not found")
        
        for index, image_annotation in enumerate(tqdm(self._data[class_name], desc=f"Writing {class_name}")):
            image = ImageProcessing.crop_image(image_annotation)

            original_image_name = image_annotation.image_path.split('/')[-2:]
            original_image_name = "_".join(original_image_name)

            image_name = self.__format_image_name(index)
            image_name = f"{image_name}_{original_image_name}"
            image_path = os.path.join(destination_path, image_name)
            image_path = format_path(image_path)

            cv2.imwrite(image_path, image)


    def __organize_class(self, class_name: str):
        images_path = os.path.join(self._data_path, class_name)
        images_path = format_path(images_path)

        if not os.path.exists(images_path):
            raise FileNotFoundError(f"Directory {images_path} not found")
        
        annotaions_path = os.path.join(self._data_path, f"{class_name}_annotations.txt")
        annotaions_path = format_path(annotaions_path)

        if not os.path.exists(annotaions_path):
            raise FileNotFoundError(f"File {annotaions_path} not found")
        
        with open(annotaions_path, 'r') as file:
            lines = file.readlines()

        for line in tqdm(lines, desc=f"Organizing {class_name}"):
            line = line.strip()
            image_name, xmin, ymin, xmax, ymax, annotation = line.split(' ')


            image_path = os.path.join(images_path, image_name)
            image_path = format_path(image_path)

            bounding_box = BoundingBox(int(xmin), int(ymin), int(xmax), int(ymax))
            image_annotation = ImageAnnotation(image_path, bounding_box)

            self._data[annotation].append(image_annotation)


if __name__ == "__main__":
    data_path = "../data/train"
    destination_path = "../data/cropped_train"
    data_organizer = DataOrganizer(data_path, destination_path)
    data_organizer.organize_data()