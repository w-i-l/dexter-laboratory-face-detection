import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple

class Evaluation:
    def __init__(self):
        pass
        
    def _compute_iou(self, bbox_a: List[int], bbox_b: List[int]) -> float:
        """Compute Intersection over Union between two bounding boxes"""
        x_a = max(bbox_a[0], bbox_b[0])
        y_a = max(bbox_a[1], bbox_b[1])
        x_b = min(bbox_a[2], bbox_b[2])
        y_b = min(bbox_a[3], bbox_b[3])
        
        # Compute intersection area
        inter_area = max(0, x_b - x_a + 1) * max(0, y_b - y_a + 1)
        
        # Compute union area
        box_a_area = (bbox_a[2] - bbox_a[0] + 1) * (bbox_a[3] - bbox_a[1] + 1)
        box_b_area = (bbox_b[2] - bbox_b[0] + 1) * (bbox_b[3] - bbox_b[1] + 1)
        
        iou = inter_area / float(box_a_area + box_b_area - inter_area)
        return iou
    
    def _compute_average_precision(self, recall: np.ndarray, precision: np.ndarray) -> float:
        """Compute Average Precision using the 11-point interpolation"""
        # Add sentinel values to begin and end
        mrec = np.concatenate(([0.], recall, [1.]))
        mpre = np.concatenate(([0.], precision, [0.]))
        
        # Make precision monotonically decreasing
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = max(mpre[i - 1], mpre[i])
            
        # Find points where recall changes
        i = np.where(mrec[1:] != mrec[:-1])[0] + 1
        
        # Sum ∆recall * precision
        ap = np.sum((mrec[i] - mrec[i - 1]) * mpre[i])
        return ap
    
    def evaluate(self, predictions: List[Tuple[str, List[int], float]], 
                ground_truth: List[Tuple[str, List[int], str]], 
                iou_threshold: float = 0.3,
                character_name: str = "Character") -> Tuple[float, np.ndarray, np.ndarray]:
        """
        Evaluate predictions against ground truth
        
        Args:
            predictions: List of (image_name, bbox, confidence_score)
            ground_truth: List of (image_name, bbox, class_name)
            iou_threshold: IoU threshold for considering a detection as correct
            character_name: Name of the character for plot title
            
        Returns:
            average_precision: The computed AP value
            recall: Array of recall values
            precision: Array of precision values
        """
        # Sort predictions by confidence score in descending order
        predictions = sorted(predictions, key=lambda x: x[2], reverse=True)
        
        num_gt = len(ground_truth)
        num_pred = len(predictions)
        
        # Initialize arrays to keep track of true/false positives
        true_positives = np.zeros(num_pred)
        false_positives = np.zeros(num_pred)
        gt_matched = np.zeros(num_gt)  # Keep track of matched ground truth boxes
        
        # For each prediction
        for pred_idx, (pred_img_name, pred_bbox, _) in enumerate(predictions):
            max_iou = -1
            max_gt_idx = -1
            
            # Find the best matching ground truth box
            for gt_idx, (gt_img_name, gt_bbox, _) in enumerate(ground_truth):
                if pred_img_name != gt_img_name:
                    continue
                    
                iou = self._compute_iou(pred_bbox, gt_bbox)
                if iou > max_iou:
                    max_iou = iou
                    max_gt_idx = gt_idx
            
            # Classify the prediction as TP or FP
            if max_iou >= iou_threshold:
                if not gt_matched[max_gt_idx]:
                    true_positives[pred_idx] = 1
                    gt_matched[max_gt_idx] = 1
                else:
                    false_positives[pred_idx] = 1
            else:
                false_positives[pred_idx] = 1
        
        # Compute cumulative values
        cum_tp = np.cumsum(true_positives)
        cum_fp = np.cumsum(false_positives)
        
        # Compute recall and precision
        recall = cum_tp / num_gt
        precision = cum_tp / (cum_tp + cum_fp)
        
        # Compute average precision
        ap = self._compute_average_precision(recall, precision)
        
        # Plot precision-recall curve
        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, '-')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(f'{character_name} Detection: AP = {ap:.3f}')
        plt.grid(True)
        plt.show()
        
        return ap, recall, precision
    

if __name__ == "__main__":
    # Example usage
    evaluator = Evaluation()
    
    class_name = 'all'
    dataset_type = 'validation'

    # Example predictions and ground truth
    predictions = []
    predictions_path = f"../data/predictions/{class_name}_predictions.txt"
    with open(predictions_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            img_name = parts[0]
            bbox = list(map(int, parts[1:5]))
            conf_score = float(parts[5])
            predictions.append((img_name, bbox, conf_score))

    # predictions = predictions[:21]
    
    last_image = predictions[-1][0]
    last_image_index = int(last_image.split(".")[0])

    ground_truth = []
    gt_path = f"../data/{dataset_type}/{class_name}_annotations.txt"
    with open(gt_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            img_name = parts[0]
            bbox = list(map(int, parts[1:5]))
            _class_name = parts[5]

            image_index = int(img_name.split(".")[0])
            if image_index > last_image_index:
                break

            ground_truth.append((img_name, bbox, _class_name))

    # Evaluate predictions
    ap, recall, precision = evaluator.evaluate(predictions, ground_truth, iou_threshold=0.3, character_name=class_name)
    print(f"Average Precision: {ap:.3f}")
    print(f"Recall: {recall}")
    print(f"Precision: {precision}")
