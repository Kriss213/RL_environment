import numpy as np

class Position:
    def __init__(self, x: float, y: float, theta: float):
        self.x = x # meters
        self.y = y # meters
        self.theta = theta # radians

    def __call__(self, *args, **kwds) -> np.ndarray:
        return np.array([
            self.x,
            self.y,
            self.theta
        ])
    
# test
if __name__ == "__main__":
    # test
    a = Position(1.0, 2.0, 3.0)
    print(a())