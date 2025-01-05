from utils.helper_functions import format_path
import csv
import numpy as np
from typing import Callable
from face_detection.cnn_model import CNNModel
from tqdm import tqdm

class ImageSlider:
    def __init__(self, clusters_path: str):
        self.__initialize(clusters_path)


    def __initialize(self, clusters_path: str):
        self._clusters_path = format_path(clusters_path)
        self._clusters: list[tuple[float, float]] = []

        with open(self._clusters_path, 'r') as file:
            reader = csv.reader(file)
            next(reader) # Skip header

            for row in reader:
                x, y = row
                self._clusters.append((float(x), float(y)))


    def get_clusters(self) -> list[tuple[float, float]]:
        return self._clusters
    

    def slide_image(
        self,
        image: np.ndarray,
        callback: Callable[[np.ndarray], float],
        tqdm: Callable[[list, str], list]
    ) -> list[np.ndarray]:
        
        faces = []

        for cluster_size in tqdm(reversed(self._clusters), desc="Iterating over clusters"):
            cluster_width, cluster_height = cluster_size
            cluster_width = int(cluster_width)
            cluster_height = int(cluster_height)
            image_height, image_width, _ = image.shape

            # setting stride 10% of the cluster size
            stride_x = int(cluster_width * 0.1)
            stride_y = int(cluster_height * 0.1)

            for y in tqdm(range(0, int(image_height - cluster_height), stride_y), desc="Iterating over y"):
                for x in range(0, int(image_width - cluster_width), stride_x):
                    cropped_image = image[y:y+cluster_height, x:x+cluster_width]
                    # cv2.imshow("Cropped image", cropped_image)
                    # cv2.waitKey(0)
                    # cv2.destroyAllWindows()
                    prediction = callback(cropped_image)
                    if prediction >= 0.5:
                        bounding_box = (x, y, x+cluster_width, y+cluster_height)
                        faces.append(bounding_box)

        return faces
    

    def print_debug_info(self):
        image_height, image_width = 480, 360
        
        # Calculate maximum widths for each column for proper alignment
        max_cluster_width = max(len(f"{int(w)}x{int(h)}") for w, h in self._clusters) + 10
        max_y_width = len("Y Iterations") + 5
        max_x_width = len("X Iterations") + 5

        # Create format string for consistent column widths
        row_format = f"│ {{:<{max_cluster_width}}} │ {{:^{max_y_width}}} │ {{:^{max_x_width}}} │"
        separator = f"├{'─' * (max_cluster_width + 2)}┼{'─' * (max_y_width + 2)}┼{'─' * (max_x_width + 2)}┤"
        top_border = f"┌{'─' * (max_cluster_width + 2)}┬{'─' * (max_y_width + 2)}┬{'─' * (max_x_width + 2)}┐"
        bottom_border = f"└{'─' * (max_cluster_width + 2)}┴{'─' * (max_y_width + 2)}┴{'─' * (max_x_width + 2)}┘"

        # Print header
        print(f"\nImage Resolution: {image_width} x {image_height}")
        print(top_border)
        print(row_format.format("Cluster Size", "Y Iterations", "X Iterations"))
        print(separator)

        # Print data rows
        total_iterations = 0
        for cluster_size in reversed(self._clusters):
            cluster_width, cluster_height = map(int, cluster_size)
            stride_x = max(1, int(cluster_width * 0.1))
            stride_y = max(1, int(cluster_height * 0.1))
            
            y_iters = len(range(0, int(image_height - cluster_height), stride_y))
            x_iters = len(range(0, int(image_width - cluster_width), stride_x))
            total_iterations += y_iters * x_iters
            
            print(row_format.format(
                f"{cluster_width} x {cluster_height}",
                str(y_iters),
                str(x_iters)
            ))
        
        # Print footer
        print(bottom_border)
        print(f"\nTotal window iterations: {total_iterations:,}")


    def slide_image_batch(
        self,
        image: np.ndarray,
        model: CNNModel,
        batch_size: int = 64,
        stride_factor: float = 0.3,
        face_threshold: float = 0.7,
    ) -> list[tuple[tuple[int, int, int, int], float]]:
        """
        Slide over image using batched predictions for better performance
        """
        faces = []
        image_height, image_width, _ = image.shape

        for cluster_size in tqdm(reversed(self._clusters), desc="Processing clusters"):
            cluster_width, cluster_height = map(int, cluster_size)
            
            stride_x = max(1, int(cluster_width * stride_factor))
            stride_y = max(1, int(cluster_height * stride_factor))

            # Collect windows for this cluster size
            windows = []
            window_positions = []

            for y in tqdm(range(0, image_height - cluster_height, stride_y), 
                         desc=f"Scanning rows (cluster {cluster_width}x{cluster_height})", 
                         leave=False):
                for x in range(0, image_width - cluster_width, stride_x):
                    # Extract window
                    window = image[y:y+cluster_height, x:x+cluster_width]
                    windows.append(window)
                    window_positions.append((x, y))

                    # When we have enough windows, process the batch
                    if len(windows) >= batch_size:
                        # Convert to batch array and get predictions
                        batch = np.array(windows)
                        predictions = model.predict_batch(batch)
                        
                        # Add detected faces
                        for i, pred in enumerate(predictions):
                            if pred >= face_threshold:
                                x, y = window_positions[i]
                                bbox = (x, y, x + cluster_width, y + cluster_height)
                                faces.append((bbox, pred))
                        
                        # Clear for next batch
                        windows = []
                        window_positions = []

            # Process any remaining windows
            if windows:
                batch = np.array(windows)
                predictions = model.predict_batch(batch)

                # if there is only one prediction
                if isinstance(predictions, float):
                    if predictions >= face_threshold:
                        x, y = window_positions[0]
                        bbox = (x, y, x + cluster_width, y + cluster_height)
                        faces.append((bbox, predictions)) 

                # if there are multiple predictions
                else:    
                    for i, pred in enumerate(predictions):
                        if pred >= face_threshold:
                            x, y = window_positions[i]
                            bbox = (x, y, x + cluster_width, y + cluster_height)
                            faces.append((bbox, pred))

        return faces
    

if __name__ == "__main__":
    image_slider = ImageSlider("../data/clusters/kmeans_clusters.csv")
    image_slider.print_debug_info()