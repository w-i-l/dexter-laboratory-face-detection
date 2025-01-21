from pipelines.detection_pipeline import DetectionPipeline
from models.detection_params import DetectionParams
from pipelines.recognition_pipeline import RecognitionPipeline
from utils.image_slider import ImageSlider
from face_detection.cnn_model import CNNModel as DetectionModel
from face_recognition.cnn_model import CNNModel as RecognitionModel
from utils.image_processing import ImageProcessing
import os

class MainPipeline:
    def __init__(
        self,
        detection_model_path: str,
        clusters_path: str,
        recognition_model_path: str,
    ):
        input_shape = (ImageProcessing.IMAGE_WIDTH, ImageProcessing.IMAGE_HEIGHT, 3)
        print("Loading detection model...")
        self._detection_model = DetectionModel(input_shape, verbose=False)
        self._detection_model.load_model(detection_model_path)
        print("Loading recognition model...")
        self._recognition_model = RecognitionModel(input_shape, verbose=False)
        self._recognition_model.load_model(recognition_model_path)
        self._image_slider = ImageSlider(clusters_path)


    def process(self, images_path: str, output_path: str, detection_params: DetectionParams):
        detection_pipeline = DetectionPipeline(images_path, detection_params, self._detection_model, self._image_slider)
        detections = detection_pipeline.process()
        output_path = output_path + "/task1"
        detection_pipeline.write_predictions(detections, output_path)

        recognition_pipeline = RecognitionPipeline(self._recognition_model, output_path, images_path)
        detections = recognition_pipeline.process()
        print(output_path)
        output_path = output_path.replace("task1", "task2")
        print(output_path)

        recognition_pipeline.write_detections(detections, output_path)