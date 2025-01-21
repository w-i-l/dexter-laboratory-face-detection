from pipelines.main_pipeline import MainPipeline
from models.detection_params import DetectionParams
import os

if __name__ == "__main__":
    # turn off tensorflow logging
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    detection_model_path = "../models/face_detector.h5"
    clusters_path = "../data/clusters/good_kmeans_clusters.csv"
    recognition_model_path = "../models/face_recognizer.h5"
    images_path = "../data/test"
    output_path = "../data/output"
    detection_params = DetectionParams(
        batch_size=1024,
        stride_factor=0.3,
        face_threshold=0.8,
        aggregator_threshold=0.18
    )

    pipeline = MainPipeline(detection_model_path, clusters_path, recognition_model_path)
    pipeline.process(images_path, output_path, detection_params)