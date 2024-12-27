class BoundingBox:
    def __init__(self, xmin: int, ymin: int, xmax: int, ymax: int):
        self.xmin = xmin
        self.ymin = ymin
        self.xmax = xmax
        self.ymax = ymax


    def __repr__(self):
        return f"BoundingBox(xmin={self.xmin}, ymin={self.ymin}, xmax={self.xmax}, ymax={self.ymax})"