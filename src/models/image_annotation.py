from models.bounding_box import BoundingBox

class ImageAnnotation:
    def __init__(self, image_path: str, bounding_box: BoundingBox):
        self.image_path = image_path
        self.bounding_box = bounding_box

    
    def __repr__(self):
        return f"Annotation(image_path={self.image_path}, bounding_box={self.bounding_box})"

