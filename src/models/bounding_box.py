class BoundingBox:
    def __init__(self, xmin: int, ymin: int, xmax: int, ymax: int):
        self.xmin = xmin
        self.ymin = ymin
        self.xmax = xmax
        self.ymax = ymax


    def get_coordinates(self) -> tuple[int, int, int, int]:
        return self.xmin, self.ymin, self.xmax, self.ymax


    def get_area(self) -> int:
        return (self.xmax - self.xmin) * (self.ymax - self.ymin)
    

    def __repr__(self):
        return f"BoundingBox(xmin={self.xmin}, ymin={self.ymin}, xmax={self.xmax}, ymax={self.ymax})"