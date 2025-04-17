import numpy as np

class Position:
    def __init__(self, x: float, y: float, theta: float):
        self.x = x # meters
        self.y = y # meters
        self._theta = None # radians
        self.theta = theta # radians


    @property
    def theta(self) -> float:
        return self._theta
    
    @theta.setter
    def theta(self, value: float):
        # clip theta to [-pi, pi]
        self._theta = value % (2 * np.pi) - np.pi

    def __call__(self, *args, **kwds) -> np.ndarray:
        return np.array([
            self.x,
            self.y,
            self.theta
        ])
    
    def __repr__(self):
        return f"Position(x={self.x:.2f}, y={self.y:.2f}, theta={self.theta:.2f})"
    
# test
if __name__ == "__main__":
    # test
    a = Position(1.0, 2.0, 3.0)
    print(a)