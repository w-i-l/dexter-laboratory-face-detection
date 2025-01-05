class NMSAggregator:
    
    @staticmethod
    def get_face_boxes(boxes, scores, iou_threshold=0.5):
        """
        Apply Non-Maximum Suppression to remove overlapping boxes
        
        Args:
            boxes: List of boxes in format [x1, y1, x2, y2]
            scores: List of confidence scores for each box
            iou_threshold: IoU threshold for suppression
            
        Returns:
            List of boxes after NMS
        """
        if not boxes:
            return []
        
        # Convert to list of tuples for easier sorting
        box_scores = list(zip(boxes, scores))
        
        # Sort by score in descending order
        box_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Separate back into boxes and scores
        sorted_boxes = [box for box, _ in box_scores]
        sorted_scores = [score for _, score in box_scores]
        
        # Keep track of which boxes to keep
        keep_boxes = []
        
        # Process boxes in order of decreasing confidence
        while len(sorted_boxes) > 0:
            # Keep the current highest scoring box
            current_box = sorted_boxes[0]
            keep_boxes.append(current_box)
            
            if len(sorted_boxes) == 1:
                break
                
            # Remove the current box
            sorted_boxes = sorted_boxes[1:]
            sorted_scores = sorted_scores[1:]
            
            # Calculate IoU with remaining boxes
            remove_indices = []
            for i, box in enumerate(sorted_boxes):
                if NMSAggregator.__compute_overlap(current_box, box, sorted_scores[i]) > iou_threshold:
                    remove_indices.append(i)
            
            # Remove boxes with high IoU
            sorted_boxes = [box for i, box in enumerate(sorted_boxes) if i not in remove_indices]
            sorted_scores = [score for i, score in enumerate(sorted_scores) if i not in remove_indices]
        
        return keep_boxes
    
    @staticmethod
    def __compute_overlap(box1, box2, score, weighted=True):
        """
        Compute IoU between two boxes, optionally weighted by confidence score
        """
        # Calculate intersection coordinates
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        # Calculate areas
        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - intersection
        
        iou = intersection / union if union > 0 else 0
        
        # Optionally weight IoU by confidence score
        if weighted:
            iou *= score
            
        return iou