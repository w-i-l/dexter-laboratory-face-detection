class DetectionParams:
    def __init__(
            self,
            batch_size: int = 1024,
            stride_factor: float = 0.3,
            face_threshold: float = 0.8,
            aggregator_threshold: float = 0.18
    ):
        self.batch_size = batch_size
        self.stride_factor = stride_factor
        self.face_threshold = face_threshold
        self.aggregator_threshold = aggregator_threshold