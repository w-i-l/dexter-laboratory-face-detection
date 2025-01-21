from face_recognition.cnn_model import CNNModel
import cv2
import os
from utils.image_processing import ImageProcessing
import numpy as np
from tqdm import tqdm

class RecognitionPipeline:
    def __init__(self, model: CNNModel, task1_path: str, images_path: str):
        self._model = model
        self._images_path = images_path
        self._task1_path = task1_path
        self._predictions = self.__read_predictions()


    def __read_predictions(self):
        detections_file_path = os.path.join(self._task1_path, "detections_all_faces.npy")
        file_names_file_path = os.path.join(self._task1_path, "file_names_all_faces.npy")
        scores_file_path = os.path.join(self._task1_path, "scores_all_faces.npy")

        detections = np.load(detections_file_path)
        file_names = np.load(file_names_file_path)
        scores = np.load(scores_file_path)

        predictions = {}
        for bbox, file_name, score in zip(detections, file_names, scores):
            if file_name not in predictions:
                predictions[file_name] = []
            predictions[file_name].append((bbox, score))

        return predictions


    def process(self, batch_size: int = 1024):
        batch = []
        batch_metadata = []

        classes = ["dad", "mom", "dexter", "deedee", "unknown"]
        detections = { class_name: [] for class_name in classes }

        for image_name in tqdm(self._predictions.keys(), desc="Recognizing faces"):
            image_path = os.path.join(self._images_path, image_name)
            image = cv2.imread(image_path)
            if image is None:
                raise FileNotFoundError(f"Image not found: {image_path}")

            faces = self._predictions[image_name]

            for bbox, score in faces:
                face = image[bbox[1]:bbox[3], bbox[0]:bbox[2]]
                face = ImageProcessing.resize_image(face)

                batch.append(face)
                batch_metadata.append((image_name, bbox))

                if len(batch) >= batch_size:
                    predictions = self._model.predict_batch(np.array(batch))

                    for (pred_class, conf), (_image_name, bbox) in zip(predictions, batch_metadata):
                        class_name = classes[pred_class]
                        detections[class_name].append((_image_name, bbox, conf))

                    batch = []
                    batch_metadata = []


        if len(batch) > 0:
            predictions = self._model.predict_batch(np.array(batch))

            for (pred_class, conf), (image_name, bbox) in zip(predictions, batch_metadata):
                class_name = pred_class
                detections[class_name].append((image_name, bbox, conf))

        return detections
    

    def write_detections(self, detections: dict[str, list[tuple[str, tuple[int, int, int, int], float]]], output_path: str):
        for class_name in detections.keys():
            if class_name == "unknown":
                continue

            detections_file_path = os.path.join(output_path, f"detections_{class_name}.npy")
            file_names_file_path = os.path.join(output_path, f"file_names_{class_name}.npy")
            scores_file_path = os.path.join(output_path, f"scores_{class_name}.npy")

            if not os.path.exists(detections_file_path):
                try:
                    with open(detections_file_path, 'w+') as file:
                        pass
                except:
                    os.makedirs(output_path, exist_ok=True)
                    with open(detections_file_path, 'w+') as file:
                        pass

                with open(file_names_file_path, 'w+') as file:
                    pass
                with open(scores_file_path, 'w+') as file:
                    pass

            _detections = []
            file_names = []
            scores = []

            print(f"Writing predictions for {class_name} to disk...")
            for prediction in detections[class_name]:
                image_name, bbox, score = prediction
                _detections.append(bbox)
                file_names.append(image_name)
                scores.append(score)

            np.save(detections_file_path, np.array(_detections))
            np.save(file_names_file_path, np.array(file_names))
            np.save(scores_file_path, np.array(scores))

        print("Done writing predictions to disk.")



if __name__ == "__main__":
    model = CNNModel((ImageProcessing.IMAGE_WIDTH, ImageProcessing.IMAGE_HEIGHT, 3), verbose=False)
    model.load_model("../models/face_recognizer.h5")

    task1_path = "../data/Ocnaru/task1"
    images_path = "../data/validation/all"
    output_path = "../data/Ocnaru/task2"
    pipeline = RecognitionPipeline(model, task1_path, images_path)
    detections = pipeline.process() 
    pipeline.write_detections(detections, output_path)
