from models.bounding_box import BoundingBox
from models.image_annotation import ImageAnnotation
import numpy as np
import cv2

class ImageProcessing:
    @staticmethod
    def crop_image(image_annotation: ImageAnnotation) -> np.ndarray:
        image = cv2.imread(image_annotation.image_path)

        if image is None:
            raise FileNotFoundError(f"File {image_annotation.image_path} not found")

        bounding_box = image_annotation.bounding_box

        return image[bounding_box.ymin:bounding_box.ymax, bounding_box.xmin:bounding_box.xmax]