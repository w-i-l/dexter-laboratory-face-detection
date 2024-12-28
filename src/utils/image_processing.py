from models.bounding_box import BoundingBox
from models.image_annotation import ImageAnnotation
import numpy as np
import cv2

class ImageProcessing:

    IMAGE_WIDTH = 130
    IMAGE_HEIGHT = 116

    @staticmethod
    def crop_image(image_annotation: ImageAnnotation) -> np.ndarray:
        image = cv2.imread(image_annotation.image_path)

        if image is None:
            raise FileNotFoundError(f"File {image_annotation.image_path} not found")

        bounding_box = image_annotation.bounding_box

        return image[bounding_box.ymin:bounding_box.ymax, bounding_box.xmin:bounding_box.xmax]
    
    @staticmethod
    def read_image(image_path: str) -> np.ndarray:
        image = cv2.imread(image_path)

        if image is None:
            raise FileNotFoundError(f"File {image_path} not found")

        width, height = ImageProcessing.IMAGE_WIDTH, ImageProcessing.IMAGE_HEIGHT
        image = cv2.resize(image, (height, width))
        # image = np.expand_dims(image, axis=1)
        # image = np.transpose(image, (1, 0, 2))  # Transpose height and width dimensions
        return image
    

    @staticmethod
    def resize_image(image: np.ndarray) -> np.ndarray:
        width, height = ImageProcessing.IMAGE_WIDTH, ImageProcessing.IMAGE_HEIGHT
        image = cv2.resize(image, (height, width))  # cv2.resize takes (width, height)
        # image = np.expand_dims(image, axis=1)
        print("image shape", image.shape)
        return image