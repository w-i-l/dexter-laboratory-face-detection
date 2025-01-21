from faces_annotations import FacesAnnotations
import os
from models.detection_params import DetectionParams
from tqdm import tqdm
import cv2
from face_detection.cnn_model import CNNModel
from utils.image_slider import ImageSlider
from aggregators.cluster_aggregator import ClusterAggregator
from utils.image_processing import ImageProcessing
import numpy as np

class DetectionPipeline:
    def __init__(self, images_path: str, detection_params: DetectionParams, model: CNNModel, image_slider: ImageSlider):
        self._images_path = images_path
        self._detection_params = detection_params
        self._model = model
        self._image_slider = image_slider

    
    def process(self):
        predictions = {}

        for image_name in tqdm(os.listdir(self._images_path), desc="Finding faces"):
            if not self.__is_image(image_name):
                continue

            image_path = os.path.join(self._images_path, image_name)
            
            image = cv2.imread(image_path)

            faces = self._image_slider.slide_image_batch(
                image,
                self._model,
                batch_size=self._detection_params.batch_size,
                stride_factor=self._detection_params.stride_factor,
                face_threshold=self._detection_params.face_threshold,
                verbose=False
            )

            scores = [face[1] for face in faces]
            faces = [face[0] for face in faces]

            aggregated_faces = ClusterAggregator.get_face_boxes(faces, scores, iou_threshold=self._detection_params.aggregator_threshold)

            predictions[image_path] = aggregated_faces


        return predictions
    
    
    def write_predictions(self, predictions: dict[str, list[tuple[int, int, int, int]]], output_path: str):
        detections_file_path = os.path.join(output_path, "detections_all_faces.npy")    
        file_names_file_path = os.path.join(output_path, "file_names_all_faces.npy")
        scores_file_path = os.path.join(output_path, "scores_all_faces.npy")

        if not os.path.exists(detections_file_path):
            try:
                with open(detections_file_path, 'w+') as file:
                    pass
            except:
                os.makedirs(output_path)
                with open(detections_file_path, 'w+') as file:
                    pass

            with open(file_names_file_path, 'w+') as file:
                pass
            with open(scores_file_path, 'w+') as file:
                pass

        detections = []
        file_names = []
        scores = []

        print("Writing predictions to disk...")

        for image_path, predictions in predictions.items():
            for prediction in predictions:
                xmin, ymin, xmax, ymax, score = prediction

                detections.append([xmin, ymin, xmax, ymax])
                file_names.append(os.path.basename(image_path))
                scores.append(score)

        np.save(detections_file_path, np.array(detections))
        np.save(file_names_file_path, np.array(file_names))
        np.save(scores_file_path, np.array(scores))

        print("Done writing predictions to disk.")


    def sort_predictions(self, output_path: str):
        detections_file_path = os.path.join(output_path, "detections_all_faces.npy")    
        file_names_file_path = os.path.join(output_path, "file_names_all_faces.npy")
        scores_file_path = os.path.join(output_path, "scores_all_faces.npy")

        detections = np.load(detections_file_path)
        file_names = np.load(file_names_file_path)
        scores = np.load(scores_file_path)

        # sort by file name
        sorted_indices = np.argsort(file_names)
        detections = detections[sorted_indices]
        file_names = file_names[sorted_indices]
        
        scores = scores[sorted_indices]

        np.save(detections_file_path, detections)
        np.save(file_names_file_path, file_names)
        np.save(scores_file_path, scores)

        
    def __is_image(self, filename: str) -> bool:
        return filename.endswith(".jpg") or filename.endswith(".png")


if __name__ == "__main__":
    detection_params = DetectionParams()
    images_path = "../data/validation/all"
    model = CNNModel((ImageProcessing.IMAGE_WIDTH, ImageProcessing.IMAGE_HEIGHT, 3))
    model.load_model("../models/face_detector.h5")
    image_slider = ImageSlider("../data/clusters/good_kmeans_clusters.csv")
    detection_pipeline = DetectionPipeline(images_path, detection_params, model, image_slider)
    predictions = detection_pipeline.process()
    detection_pipeline.write_predictions(predictions, "../data/Ocnaru/task1")
    detection_pipeline.sort_predictions("../data/Ocnaru/task1")



