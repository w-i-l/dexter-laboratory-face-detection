from utils.helper_functions import compute_iou

class ClusterAggregator:

    @staticmethod
    def get_face_boxes(boxes, scores, iou_threshold=0.25):
        """
        Main function to cluster boxes and return merged boxes for each cluster
        """
        # Find clusters of overlapping boxes
        clusters = ClusterAggregator.__cluster_boxes(boxes, iou_threshold)
        
        # Merge boxes in each cluster
        # merged_boxes = [merge_boxes_in_cluster(boxes, cluster) for cluster in clusters]
        merged_boxes = [ClusterAggregator.__merge_boxes_in_cluster(boxes, cluster, scores) for cluster in clusters]
        
        return merged_boxes
    

    @staticmethod
    def __cluster_boxes(boxes, iou_threshold=0.5):
        """
        Cluster boxes based on IoU overlap.
        Returns a list of clusters, where each cluster is a list of box indices.
        """
        if not boxes:
            return []
        
        # Initialize clusters with each box in its own cluster
        clusters = [[i] for i in range(len(boxes))]
        
        # Merge clusters iteratively
        merged = True
        while merged:
            merged = False
            i = 0
            while i < len(clusters):
                j = i + 1
                while j < len(clusters):
                    # Check if any box in cluster i overlaps significantly with any box in cluster j
                    merge_clusters = False
                    for idx1 in clusters[i]:
                        for idx2 in clusters[j]:
                            if compute_iou(boxes[idx1], boxes[idx2]) > iou_threshold:
                                merge_clusters = True
                                break
                        if merge_clusters:
                            break
                    
                    if merge_clusters:
                        # Merge cluster j into cluster i
                        clusters[i].extend(clusters[j])
                        clusters.pop(j)
                        merged = True
                    else:
                        j += 1
                i += 1
        
        return clusters
    

    @staticmethod
    def __merge_boxes_in_cluster(boxes, cluster_indices, scores):
        """
        Merge boxes in a cluster by taking the average of coordinates
        """
        cluster_boxes = [boxes[i] for i in cluster_indices]
        scores = [scores[i] for i in cluster_indices]

        # # Simple average
        # x1 = sum(box[0] for box in cluster_boxes) / len(cluster_boxes)
        # y1 = sum(box[1] for box in cluster_boxes) / len(cluster_boxes)
        # x2 = sum(box[2] for box in cluster_boxes) / len(cluster_boxes)
        # y2 = sum(box[3] for box in cluster_boxes) / len(cluster_boxes)

        if sum(scores) == 0:
            return 0, 0, 0, 0

        # ponderate average
        x1 = sum(box[0] * scores[i] for i, box in enumerate(cluster_boxes)) / sum(scores)
        y1 = sum(box[1] * scores[i] for i, box in enumerate(cluster_boxes)) / sum(scores)
        x2 = sum(box[2] * scores[i] for i, box in enumerate(cluster_boxes)) / sum(scores)
        y2 = sum(box[3] * scores[i] for i, box in enumerate(cluster_boxes)) / sum(scores)

        return int(x1), int(y1), int(x2), int(y2)